import torch
from node import Node, ModelConfig
from worker_nano_vllm import nanovLLMWorker
import torch.multiprocessing as mp
import queue
import yaml
import time
from request import ExecuteInfo, Request
from logging import getLogger

logger = getLogger(__name__)
logger.setLevel("INFO")

class optimizer:
    def __init__(self, config, **kwargs):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config
        mp.set_start_method("spawn", force=True)
        
        self.device_cnt = torch.cuda.device_count()
        self.processes = []
        self.cmd_queues = []
        self.result_queues = []
        
        self.create_nodes(config)
        self.create_workers()
        logger.info("optimizer initialized")

    def create_nodes(self, config):
        num_nodes = len(config['nodes'])
        self.nodes = {
            f'node{i}': Node(id=f'node{i}') for i in range(num_nodes)
        }
        
        self.models = set()
        for node_name, node in self.nodes.items():
            node.input_nodes = [
                self.nodes[input_node_id] for input_node_id in config['nodes'][node_name]['input_nodes']
            ]
            node.output_nodes = [
                self.nodes[output_node_id] for output_node_id in config['nodes'][node_name]['output_nodes']
            ]
            
            model = config['nodes'][node_name]['model']
            self.models.add(model)
            
            node.model_config = ModelConfig(
                model_name=model,
                system_prompt=config['nodes'][node_name].get('prompt'),
                temperature=config['nodes'][node_name].get('temperature', 0.7),
                # top_p=config['nodes'][node_name].get('top_p', 0.9),
                max_tokens=config['nodes'][node_name].get('max_tokens', 256),
                max_batch_size=config['nodes'][node_name].get('max_batch_size', torch.inf),
                dtype=config['nodes'][node_name].get('dtype', 'bfloat16'),
                lora_config=config['nodes'][node_name].get('lora_config', None)
            )
            
        self.start_nodes = [
            self.nodes[node_name] for node_name in config['start_nodes']
        ]
        
        self.end_nodes = [
            self.nodes[node_name] for node_name in config['end_nodes']
        ]
    
    def create_workers(self):
        usable_devices = []
        for i in range(self.device_cnt):
            try:
                torch.cuda.set_device(i)
                # Try a simple CUDA call to check if device is usable
                torch.cuda.current_device()
                usable_devices.append(i)
            except Exception as e:
                logger.warning(f"Skipping CUDA device {i}: {e}")

        self.cmd_queues = []
        self.result_queues = []
        self.processes = []

        for i in usable_devices:
            cmd_queue = mp.Queue()
            result_queue = mp.Queue()
            self.cmd_queues.append(cmd_queue)
            self.result_queues.append(result_queue)
            process = mp.Process(
                target=worker_process, args=(
                    i,
                    f"cuda:{i}", 
                    cmd_queue, 
                    result_queue, 
                ),
                daemon=False
            )
            self.processes.append(process)
            process.start()
        self.device_cnt = len(usable_devices)
    
    def compute_longest_distances(self,):
        output_set = set(self.end_nodes)
        memo = {} # A memoization dictionary to store the longest distance for each node.
        # DFS to compute the longest distance for each node.
        def dfs(node):
            if node in memo:
                return memo[node]

            if node in output_set:
                memo[node] = 0
                return 0
            max_dist = -1  

            for child in node.output_nodes:
                child_dist = dfs(child)
                if child_dist != -1:  # If child is reachable.
                    max_dist = max(max_dist, child_dist + 1)
            memo[node] = max_dist
            return max_dist
        
        for node in self.nodes.values():
            node.max_distance = dfs(node)
            
        return 
    
    def optimize(self, requests):
        """
        Rerank requests by:
        1. priority descending (higher priority first)
        2. prompt length ascending (shorter prompts first when priorities tie)
        """
        # Use -x.priority for descending sort on priority
        self.requests = sorted(
            requests,
            key=lambda x: (-x.priority, x.prompt_len)
        )

    def schedule(self, requests, dp_threshold=16):
        # Initialize workflow lists and request mapping.
        self.workflows = [[] for _ in range(self.device_cnt)]
        self.optimize(requests)
        self.requests_cnt = len(self.requests)
        self.req_id_map = {req.id: req for req in self.requests}
        # Precompute the longest distances for all nodes.
        self.compute_longest_distances()

        last_nodes = []      # Nodes processed in the previous round.
        paused_nodes = []    # Nodes paused due to unsatisfied dependencies.
        device_history = {}   # Mapping from node to assigned device index.

        def partition_requests_for_node(node, requests, return_full = False):
            """
            Partition the request list for duplicate nodes.
            Assumes that duplicate_info is a tuple (dup_index, total_dup), where:
            - dup_index: the index of the current duplicate (starting from 1)
            - total_dup: total number of duplicates for the original node
            If duplicate_info is not present, return all requests.
            """
            if node.data_parallel:
                dup_index, total_dup = node.duplicate_info[0], node.duplicate_info[1] + 1
                if total_dup > 0:
                    total = len(requests)
                    per = total // total_dup
                    start = per * dup_index
                    # Last duplicate gets all remaining requests.
                    end = per * (dup_index + 1) if dup_index < total_dup else total
                    #print(f"Node {node.id} with duplicate index {dup_index} gets requests from {start} to {end}")
                    requests = requests[start:end]
            if return_full:
                return requests
            
            return [request.id for request in requests]

        while True:
            # 1. Node Generation: use start_nodes on the first round or output_nodes from last_nodes.
            if not last_nodes:
                new_nodes = list(self.start_nodes)
            else:
                new_nodes = []
                for node in last_nodes:
                    new_nodes.extend(node.output_nodes)
                # Remove duplicates (make sure nodes are hashable).
                new_nodes = list(set(new_nodes))

            # 2. Dependency Check: only nodes whose all input_nodes have been assigned a device are valid.
            valid_nodes = []
            for node in new_nodes:
                if all(input_node in device_history for input_node in node.input_nodes):
                    valid_nodes.append(node)
                    if node in paused_nodes:
                        paused_nodes.remove(node)
                else:
                    if node not in paused_nodes:
                        paused_nodes.append(node)
            new_nodes = valid_nodes

            # If no new valid nodes, try to resume paused nodes.
            if not new_nodes:
                resumed = []
                for node in paused_nodes[:]:
                    if all(input_node in device_history for input_node in node.input_nodes):
                        resumed.append(node)
                        paused_nodes.remove(node)
                new_nodes.extend(resumed)

            # Exit the loop if no new nodes are available.
            if not new_nodes:
                break

            # 3. Matching Node Count with Device Count.
            if len(new_nodes) > self.device_cnt:
                # When there are more nodes than devices, sort by max_distance (descending)
                new_nodes.sort(key=lambda x: x.max_distance, reverse=True)
                extra_nodes = new_nodes[self.device_cnt:]
                new_nodes = new_nodes[:self.device_cnt]
                for node in extra_nodes:
                    if node not in paused_nodes:
                        paused_nodes.append(node)
            elif len(new_nodes) < self.device_cnt:
                # If there are fewer nodes than devices, try to resume paused nodes.
                for node in paused_nodes[:]:
                    if all(input_node in device_history for input_node in node.input_nodes):
                        new_nodes.append(node)
                        paused_nodes.remove(node)
                        if len(new_nodes) == self.device_cnt:
                            break

                # If still fewer nodes, duplicate existing nodes.
                if len(new_nodes) < self.device_cnt and self.requests_cnt > dp_threshold:
                    required = self.device_cnt - len(new_nodes)
                    duplicates = []
                    for i in range(required):
                        # Cycle through available nodes for duplication.
                        original = new_nodes[i % len(new_nodes)]
                        duplicate = self.duplicate_node(original)
                        # If the original node does not have duplicate_info, initialize it.
                        if original.duplicate_info is None:
                            original.data_parallel = True
                            original.is_duplicate = False
                            original.duplicate_info = (0, 0)
                        # Increase the duplicate count for the original.
                        count = original.duplicate_info[1] + 1
                        original.duplicate_info = (0, count)
                        # Update the duplicate_info for all existing duplicates.
                        for node in duplicates:
                            if node.main_node == original:
                                node.duplicate_info[1] = count
                        # Set the duplicate_info for the new duplicate.
                        duplicate.duplicate_info = (count, count)
                        duplicates.append(duplicate)
                    new_nodes.extend(duplicates)

            # 4. Device Assignment: Prefer reusing a device from one of its input_nodes.
            available_devices = set(range(self.device_cnt))
            for node in new_nodes:
                assigned = False
                for input_node in node.input_nodes:
                    if input_node in last_nodes:
                        d = device_history[input_node]
                        if d in available_devices:
                            device_history[node] = d
                            available_devices.remove(d)
                            assigned = True
                            break
                if not assigned:
                    device_history[node] = available_devices.pop()

            # 6. Execute Commands: Append the execute command for each node.
            for node in new_nodes:
                req_ids = partition_requests_for_node(node, self.requests)
                self.workflows[device_history[node]].append({
                    "command": "execute",
                    "params": (node, req_ids)
                })

            # Update last_nodes for the next iteration.
            last_nodes = new_nodes

    def execute(self, requests=None, return_reqs = False, skip_exit=False):
        if requests is not None:
            start_time = time.perf_counter()
            self.schedule(requests)
            schedule_time = time.perf_counter() - start_time

        finish_flags = [False] * self.device_cnt # Flags to indicate whether a device has finished its tasks.
        worker_pointer = [0] * self.device_cnt # Pointers to the current task of each device.
        def cmd_transfer(task):  
            if task['command'] == "execute":
                node, req_ids = task['params'][0], task['params'][1]
                prompts = [self.req_id_map[req_id].prompt for req_id in req_ids]
                for req_id in req_ids:
                    prompt = self.req_id_map[req_id].prompt
                    if isinstance(prompt, list):
                        step = self.req_id_map[req_id].step
                        prompt = prompt[step]
                        
                    history_seqs = [self.req_id_map[req_id].node_output.get(inp.id, "") for inp in node.input_nodes]
                    history = "".join(history_seqs)
                    prompt = prompt + history
                    prompts.append(prompt)
                execute_info = ExecuteInfo(node, req_ids, prompts)
                task['params'] = (execute_info,)
            return task
            
        exe_start = time.perf_counter()
        for i in range(self.device_cnt):
            #print(f"Worker {i}, executing tasks: {[wf['command'] for wf in self.workflows[i]]}") # Debugging line
            if len(self.workflows[i]) > 0:
                self.cmd_queues[i].put(cmd_transfer(self.workflows[i][0]))
            else:
                finish_flags[i] = True
                if not skip_exit:
                    self.cmd_queues[i].put(("exit",()))
                
        while not all(finish_flags):
            for i in range(self.device_cnt):
                if finish_flags[i]:
                    continue
                try:
                    message = self.result_queues[i].get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if message is not None:
                    if message['command'] == "execute":
                        result = message['result']
                        node_name = result['node_name']

                        results = result['item']
                        for req in results:
                            self.req_id_map[req['id']].node_output[node_name] = req['output']
                            self.req_id_map[req['id']].step += 1
                            self.req_id_map[req['id']].benchmark[node_name] = req['benchmark']
                        benchmark = result['benchmark']
                        self.nodes[node_name].benchmark.update(benchmark)
                    
                    else:
                        pass
                        #print(f"From Worker {i} received message: {message}")

                    worker_pointer[i] += 1
                    if worker_pointer[i] < len(self.workflows[i]):
                        task = self.workflows[i][worker_pointer[i]] 
                        task = cmd_transfer(task)
                        #print(f"Worker {i} executing task: {task}") # Debugging line
                        self.cmd_queues[i].put(task)
                    else:
                        #print(f"Worker {i} finished all tasks.")
                        finish_flags[i] = True
                        if not skip_exit:
                            self.cmd_queues[i].put(("exit",()))

        # print("Execution completed.")
        if return_reqs:
            return self.requests
        return time.perf_counter() - exe_start
            
    def exit(self):
        for q in self.cmd_queues + self.result_queues:
            q.close()
            q.join_thread()
        
        for p in self.processes:
            p.join()

        logger.info("Optimizer exited")
                    
    def duplicate_node(self, node):
        new_node = Node(id=node.id)
        new_node.input_nodes = node.input_nodes
        new_node.output_nodes = []
        new_node.prompt = node.prompt
        new_node.model_config = node.model_config
        new_node.keep_cache = node.keep_cache
        new_node.data_parallel = True
        new_node.is_duplicate = True
        new_node.main_node = node
        new_node.duplicate_info = (0, 0)
        return new_node
    
    def node_reset(self):
        """
        Reset the state of all nodes.
        """
        for node in self.nodes.values():
            node.data_parallel = False
            node.duplicate_info = None

def worker_process(id, device, cmd_queue, result_queue):
    worker = nanovLLMWorker(id, device, cmd_queue, result_queue)
    worker.run()

if __name__ == '__main__':
    from request import Request
    opt = optimizer('templates/split.yaml')
    requests = [Request(_, 'What is Machine Learning System?') for _ in range(8)]
    opt.schedule(requests)
    t = opt.execute()
    print("Execution time:", t)
    for node in opt.nodes.values():
        print(f"Node {node.id} benchmark: {node.benchmark}")
    # Print generated outputs
    for req in opt.requests:
        print(f"Request {req.id}:")
        for node_name, output in req.node_output.items():
            print(f"  Node {node_name} output: {output}")
    opt.exit()