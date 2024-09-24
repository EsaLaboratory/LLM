import torch
from react import *
import argparse


def main():
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Output an LLM's answer to a question on documents")
    parser.add_argument('--model',
                        metavar='model',
                        type=str,
                        default='teknium/OpenHermes-2.5-Mistral-7B',
                        help="llm model")
    parser.add_argument('--n',
                        metavar='n',
                        type=int,
                        default=10,
                        help="Number of tests")
    parser.add_argument('--difficulty',
                        metavar='difficulty',
                        type=str,
                        default="medium",
                        help="Difficulty of test (easy, medium or hard)")
    parser.add_argument('--task0',
                        metavar='Tasks index start',
                        type=int,
                        default=0,
                        help="Tasks index of the tests (between 0 and 7)")
    parser.add_argument('--task1',
                        metavar='Tasks index end',
                        type=int,
                        default=-2,
                        help="Tasks index of the tests (between 0 and 7)")
    parser.add_argument('--react',
                        metavar='System prompt specification',
                        type=str,
                        default="react",
                        help="System prompt specification ('react', 'noreact_example', 'noreact')")

    args = parser.parse_args()
    model_name = args.model
    access_token = ""  # add your own hugging face acces token

    print("begin test")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=access_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    print('tokenizer loaded')

    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # {'': 'cuda:0'},
        quantization_config=double_quant_config,
        use_flash_attention_2=False,
        low_cpu_mem_usage=True,
        token=access_token,
    )
    print(f"model loaded in {time.time() - start}")

    n = args.n
    difficulty = args.difficulty
    task_index = (args.task0, args.task1)
    react = args.react
    print("REACT", react)
    name = model_name[model_name.find('/')+1:]
    test_type = ""
    if react == "noreact_example":
        test_type = "noreact_example_"
    elif react == "noreact":
        test_type = "noreact_"
    test_user_retriever(model, tokenizer, n, difficulty,
                        f'../data/{name}_{difficulty}_' + test_type, task_index, react)
    management_system(model,
                      tokenizer,
                      mode="easy",
                      path="../data/",
                      test=False)


if __name__ == "__main__":
    main()
