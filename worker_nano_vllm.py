import torch
import logging
import os
import queue
from request import Request
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int64":   torch.int64,
    "bool":    torch.bool,
    "bfloat16": torch.bfloat16,
}

class nanovLLMWorker:
    def __init__(
        self, 
        worker_id, 
        device_id, 
        cmd_queue, 
        response_queue
    ):
        self.id = worker_id
        self.device = device_id
        
        gpu_index = device_id.split(":")[-1]
        # Check if the device is available and supported
        try:
            gpu_index_int = int(gpu_index)
            if gpu_index_int >= torch.cuda.device_count():
                logging.warning(f"Worker {worker_id}: CUDA device index {gpu_index_int} is not available. Exiting worker.")
                # Optionally, put a message in the response queue before exiting
                if response_queue is not None:
                    response_queue.put({'command': 'exit', 'result': f'Worker {worker_id} skipped unavailable device {gpu_index_int}.'})
                exit(0)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_index_int)
                logging.info(f"Worker set to CUDA device {gpu_index}. Current device: {torch.cuda.current_device()}")
            else:
                raise RuntimeError("CUDA not available")
        except Exception as e:
            logging.warning(f"Worker {worker_id}: Exception when setting CUDA device {gpu_index}: {e}. Exiting worker.")
            if response_queue is not None:
                response_queue.put({'command': 'exit', 'result': f'Worker {worker_id} skipped device {gpu_index} due to error: {e}'})
            exit(0)
        
        self.model_name = None
        self.llm = None
        self.tokenizer = None
        self.cmd_queue = cmd_queue
        self.response_queue = response_queue
        logging.info(f'Worker {worker_id} initialized with device {self.device}')
    
    def init_node(self, node):
        self.node_name = node.id
        self.is_duplicate = node.is_duplicate
        
        config = node.model_config
        self.max_batch_size = config.max_batch_size
        self.max_gen_tokens = config.max_tokens
        self.use_chat_template = config.use_chat_template
        self.dtype = DTYPE_MAP[config.dtype]
        self.temperature = config.temperature
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_gen_tokens,
        )
        
        model_name = config.model_name
        if model_name != self.model_name:
            # switch to target model
            if self.llm is not None:
                del self.llm
                torch.cuda.empty_cache()
            if self.tokenizer is not None:
                del self.tokenizer
            self.llm = LLM(
                "/data/hf_cache/models--Qwen--Qwen3-0.6B/", # Later Refine this
                dtype=self.dtype,
                enforce_eager=True,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model_name = model_name

        # prefix caching
        self.prefix = config.system_prompt
            
    @torch.inference_mode()
    def execute(self, exe_info):
        init_start = time.perf_counter()
        self.init_node(exe_info.node)
        self.requests = [Request(id, prompt) for id, prompt in zip(exe_info.req_ids, exe_info.prompts)]
        init_time = time.perf_counter() - init_start
        
        # Process requests in batches
        prefill_start = time.perf_counter() # keep track of TTFT
        # Build batch inputs
        messages_batch = []
        for req in self.requests:
            messages = [
                {"role": "system", "content": self.prefix},
                {"role": "user",   "content": req.prompt},
            ]
            messages_batch.append(messages)

        # Apply chat template for all
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized. Please check model initialization.")
        try:
            text_inputs = []
            for messages in messages_batch:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                text_inputs.append(text)
            chat = text_inputs
        except Exception as e:
            logging.error(f"Chat template failed: {e}, using fallback")
            chats = []
            for messages in messages_batch:
                chat_text = ""
                for m in messages:
                    chat_text += f"{m['role'].capitalize()}: {m['content']}\n"
                chats.append(chat_text.strip() + "\nAssistant: ")
            chat = chats

        # Run inference
        if self.llm is None:
            raise RuntimeError("LLM model is not initialized. Please check model initialization.")
        
        # Use very conservative sampling parameters for better coherence
        sampling_params = SamplingParams(
            temperature=0.1,  # Very low temperature for more deterministic output
            max_tokens=50,    # Shorter output for testing
            ignore_eos=False
        )
        
        outputs = self.llm.generate(chat, sampling_params)
        
        results = []
        for i, output in enumerate(outputs):
            if isinstance(output, dict):
                generated_text = output.get("text", "")
            else:
                generated_text = output
            results.append({
                'id': self.requests[i].id,
                'output': generated_text,
                'benchmark': (prefill_start, time.perf_counter())
            })

        generate_time = time.perf_counter() - prefill_start
        if self.is_duplicate:
            benchmark = {'init_time': 0, 'prefill_time': 0, 'generate_time': 0}
        else:
            benchmark = {'init_time': init_time, 'prefill_time': 0.0, 'generate_time': generate_time}
               
        return {
            'item': results,
            'node_name': self.node_name,
            'benchmark': benchmark,
        }
    
    def exit(self):
        del self.llm
        del self.tokenizer
        torch.cuda.empty_cache()
        logging.info(f'Worker {self.id} exited.')
        
    def run(self, debug=True):
        while True:
            msg = self.cmd_queue.get()
            if isinstance(msg, tuple):
                command, params = msg
            elif isinstance(msg, dict):
                command = msg.get("command")
                params = msg.get("params", ())
            else:
                raise NotImplementedError

            if command == "exit":
                self.exit()
                self.response_queue.put(
                    {
                        'command': command,
                        'result': 'Worker exited.',
                        'elapsed_time': 0.0,
                    }
                    )
                break
            
            elif isinstance(command, str) and callable(getattr(self, command, None)):
                func = getattr(self, command)
                if not debug:
                    try:
                        start_time = time.perf_counter()
                        result = func(*params) if params is not None else func()
                        elapsed_time = time.perf_counter() - start_time
                        self.response_queue.put(
                            {
                                'command': command,
                                'result': result,
                                'elapsed_time': elapsed_time,
                            }
                        )
                    except Exception as e:
                        self.response_queue.put(f"error: {e}")
                else:
                    start_time = time.perf_counter()
                    result = func(*params) if params is not None else func()
                    elapsed_time = time.perf_counter() - start_time
                    self.response_queue.put(
                        {
                            'command': command,
                            'result': result,
                            'elapsed_time': elapsed_time,
                        }
                    )
            else:
                self.response_queue.put(f"Unknown: {command}")

if __name__ == "__main__":
    from request import Request,ExecuteInfo
    from node import Node, ModelConfig
    request = Request(0, 'What is the capital of France?')    
    model = 'Qwen/Qwen3-0.6B'
    
    worker = nanovLLMWorker(
        worker_id='worker_1',
        device_id="cuda:0",
        cmd_queue=queue.Queue(),
        response_queue=queue.Queue(),
    )
    
    config = ModelConfig(
        model_name=model,
        system_prompt='You are a helpful assistant.',
    )
    
    node = Node(
        id='node_1',
        model_config=config,
        keep_cache=False,
    )
    
    exe_info = ExecuteInfo(
        node=node,
        req_ids=[request.id],
        prompts=[request.prompt],
    )
    out = worker.execute(exe_info)
    print(out)
