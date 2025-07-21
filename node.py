import uuid
class Node:
    def __init__(self, id=None, prompt=None, model_config=None, keep_cache=False):
        self.id = id if id is not None else uuid.uuid4()
        self.input_nodes = []
        self.output_nodes = []
        self.prompt = prompt
        self.model_config = model_config
        self.max_distance = None
        self.keep_cache = keep_cache
        self.benchmark = Benchmark()
        
        #for data parallelism
        self.data_parallel = False
        self.is_duplicate = False
        self.main_node = None
        self.duplicate_info = None
        
class ModelConfig:
    def __init__(
        self, 
        model_name, 
        system_prompt=None, 
        temperature=0.7, 
        # top_p=0.9, 
        max_tokens=256,
        max_batch_size=8,
        dtype='bfloat16',
        use_chat_template=True,
        lora_config=None,
        ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        # self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.dtype = dtype  
        self.use_chat_template = use_chat_template
        self.lora_config = lora_config
    
class Benchmark:
    def __init__(self):
        self.init_time = 0
        self.prefill_time = 0
        self.generate_time = 0
        
    def total_time(self):
        return self.init_time + self.prefill_time + self.generate_time
    
    def update(self, dict):
        self.init_time += dict.get('init_time')
        self.prefill_time += dict.get('prefill_time')
        self.generate_time += dict.get('generate_time')
    
    def __str__(self):
        return f"Init time: {self.init_time}, Prefill time: {self.prefill_time}, Generate time: {self.generate_time}, Total time: {self.total_time()}"