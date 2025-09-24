# Licensed under the MIT license.

import os
import os
import time
from tqdm import tqdm
import concurrent.futures
# from openai import AzureOpenAI
#
# client = AzureOpenAI(
#     api_version="",
#     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
#     api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
# )
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:2427/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

max_threads = 64


def load_OpenAI_model(model):
    return None, model


def generate_with_OpenAI_model(
    prompt,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
):
    messages = [{"role": "user", "content": prompt}]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
    }

    ans, timeout = "", 5
    while not ans:
        try:
            time.sleep(timeout)
            completion = client.chat.completions.create(messages=messages, **parameters)
            ans = completion.choices[0].message.content

        except Exception as e:
            print(e)
        if not ans:
            timeout = timeout * 2
            if timeout > 120:
                timeout = 1
            try:
                print(f"Will retry after {timeout} seconds ...")
            except:
                pass
    return ans


def generate_n_with_OpenAI_model(
    prompt,
    n=1,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=[],
    max_threads=3,
    disable_tqdm=True,
):
    preds = []

    messages = [{"role": "user", "content": prompt}]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
    }

    ans, timeout = [], 5
    while not ans:
        try:
            time.sleep(timeout)
            completion = client.completions.create(prompt=prompt, **parameters, n=n, echo=False)
            ans = [completion.choices[i].text for i in range(len(completion.choices))]
            # print(ans)
            # print('='*10)
            

        except Exception as e:
            print(e)
        if not ans:
            timeout = timeout * 2
            if timeout > 120:
                timeout = 1
            try:
                print(f"Will retry after {timeout} seconds ...")
            except:
                pass
    return ans
