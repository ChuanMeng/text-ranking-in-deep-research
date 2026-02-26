import argparse
import math
from typing import List, Tuple
import torch
import torch.nn.functional as F  
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from vllm import LLM, SamplingParams
import json
import pandas as pd
from tqdm import tqdm
import os
import logging
from openai import OpenAI

import threading


def add_ranks(df, sort_by='score', ascending=False, groupby='query_id'):
    df = df.sort_values(by=[groupby, sort_by], ascending=[True, ascending])
    df['rank'] = df.groupby(groupby).cumcount() + 1
    return df


class Rank1:
    def __init__(
        self,
        model: str,
        context_size: int = 450,
        device: str = "cuda",
        batch_size: int = 32,
        precision: str = "float16",
        vllm_batched: bool = True,  
        max_output_tokens: int = 2000,
        num_gpus: int = 1,
        text_field='text',
        api_url: str = None
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._context_size = context_size
        self._device = device
        self._precision = precision
        self._vllm_batched = vllm_batched 
        self._max_output_tokens = max_output_tokens
        self.use_classifier = False
        self.batch_size = batch_size
        self.text_field = text_field
        self.fix_exception_count = 0
        self.fix_call_count = 0
        self._api_url = api_url
        
        # Initialize OpenAI client if using API
        if self._api_url is not None:
            self._openai_client = OpenAI(base_url=api_url, api_key="EMPTY")

        # left padding
        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token

        self.true_token = self._tokenizer(" true", add_special_tokens=False).input_ids[0]
        self.false_token = self._tokenizer(" false", add_special_tokens=False).input_ids[0]
        self.think_token = self._tokenizer("<think>", add_special_tokens=False).input_ids[0]
        self.think_end_token = self._tokenizer("</think>", add_special_tokens=False).input_ids[-1]

        # Only initialize model/engine if not using API
        if self._api_url is None:
            if self._vllm_batched:
                self._vllm_engine = LLM(
                    model=model,
                    tensor_parallel_size=1,
                    dtype=precision,
                    trust_remote_code=True,
                    max_model_len=4000,
                    gpu_memory_utilization=0.9,
                    #
                    enforce_eager=True,
                    compilation_config={"level": 1},
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16 if precision == "float16" else torch.float32,
                    device_map=device,
                )
                self._model.eval()

        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_output_tokens,
            logprobs=20,
            stop=["</think> true", "</think> false"],
            skip_special_tokens=False,
        )
        self._generation_config = GenerationConfig(
            temperature=0,
            max_new_tokens=max_output_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )


    def return_prompt(self, query, doc_content, is_plain: bool = True) -> str:
        return "Determine if the following passage is relevant to the query. " \
                "Answer only with 'true' or 'false'.\n" \
                f"Query: {query}\n" \
                f"Passage: {doc_content}\n" \
                "<think>" # force the model to start with this

    def _truncate_by_context(self, prompt: str) -> str:
        encoded = self._tokenizer.encode(prompt)
        if len(encoded) > self._context_size:
            prompt = self._tokenizer.decode(encoded[: self._context_size - 10])
        return prompt

    def _fix_incomplete_responses(
        self, original_prompts: List[str], generated_texts: List[str]
    ) -> Tuple[List[str], List[int], List[float]]:

        cleaned_texts = []
        for text in generated_texts:
            text = text.rstrip()

            if not text.endswith(('.', '!', '?')): # remove incomplete sentences
                last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last_punct != -1:
                    text = text[: last_punct + 1]
            cleaned_texts.append(text.strip())

        forced_prompts = [
            f"{original_prompt}\n{cleaned_text}\n</think>"
            for original_prompt, cleaned_text in zip(original_prompts, cleaned_texts)
        ]

        new_sampling_args = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
            skip_special_tokens=False,
        )
        outputs = self._vllm_engine.generate(forced_prompts, new_sampling_args, use_tqdm=False)

        all_final_texts, all_token_counts, all_scores = [], [], []

        for i in range(len(outputs)):
            try:
                text = outputs[i].outputs[0].text
                final_logits = outputs[i].outputs[0].logprobs[-1]
                assert (
                    self.false_token in final_logits and self.true_token in final_logits
                ), f"final logits are missing true or false: {final_logits}"
            except Exception as e:
                print(f"Error: {e} on fixing error, setting at 0.5 score: {outputs[i].outputs}")
                self.fix_exception_count += 1
                
                all_final_texts.append(text)
                all_token_counts.append(len(outputs[i].outputs[0].token_ids))
                all_scores.append(0.5)
                
                continue

            token_count = len(outputs[i].outputs[0].token_ids)
            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob

            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            p_true = true_score / (true_score + false_score)
        
            all_final_texts.append(text)
            all_token_counts.append(token_count)
            all_scores.append(p_true)

        return all_final_texts, all_token_counts, all_scores

    def _call_api_batched(self, prompts: List[str]):
        """Call the completions API for batched prompts."""
        # Tokenize prompts first
        tokenized_prompts = []
        for prompt in prompts:
            token_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
            tokenized_prompts.append(token_ids)
        
        try:
            # Call OpenAI completions API
            response = self._openai_client.completions.create(model="jhu-clsp/rank1-7b", prompt=prompts, temperature=0, max_tokens=self._max_output_tokens, logprobs=20, stop=["</think> true", "</think> false"], extra_body={"skip_special_tokens": False})
            # Convert response to dict for compatibility
            result = response.model_dump()
            # import pdb; pdb.set_trace()
        except Exception as e:
            print(f"Error calling API: {e}")
            # import pdb; pdb.set_trace()
            raise
        
        # Parse response to match vLLM output format
        # The API should return a structure similar to OpenAI's completions API
        # with choices[].text, choices[].logprobs, etc.
        all_outputs = [None] * len(prompts)
        all_output_token_counts = [None] * len(prompts)
        all_scores = [None] * len(prompts)
        incomplete_prompts, incomplete_texts, incomplete_indices = [], [], []
        
        if "choices" in result:
            for i, choice in enumerate(result["choices"]):
                text = choice.get("text", "")
                
                # Get logprobs - handle different API response formats
                logprobs_data = choice.get("logprobs", {})
                
                # Try to extract the final token's logprobs
                try:
                    token_logprobs = logprobs_data["token_logprobs"]
                    tokens = logprobs_data["tokens"]
                    
                    # Get top_logprobs for the last token to find true/false probabilities
                    top_logprobs = logprobs_data.get("top_logprobs", [])
                    if top_logprobs and len(top_logprobs) > 0:
                        final_logprobs_dict = top_logprobs[-1]
                    else:
                        final_logprobs_dict = {}
                    
                    # Convert token strings to token IDs for comparison
                    true_token_str = self._tokenizer.decode([self.true_token])
                    false_token_str = self._tokenizer.decode([self.false_token])
                    
                    # Look for true/false in the final logprobs
                    true_logit = None
                    false_logit = None
                    
                    for token_key, logprob in final_logprobs_dict.items():
                        if token_key == true_token_str:
                            true_logit = logprob
                        elif token_key == false_token_str:
                            false_logit = logprob
                    
                    if true_logit is None or false_logit is None:
                        # Incomplete response, needs fixing
                        incomplete_prompts.append(prompts[i])
                        incomplete_texts.append(text)
                        incomplete_indices.append(i)
                        continue
                    
                    # Calculate probability
                    true_score = math.exp(true_logit)
                    false_score = math.exp(false_logit)
                    p_true = true_score / (true_score + false_score)
                    
                    # Count tokens
                    if tokens:
                        token_count = len(tokens)
                    else:
                        # Fallback: tokenize the text
                        token_count = len(self._tokenizer.encode(text))
                    
                    all_outputs[i] = text
                    all_output_token_counts[i] = token_count
                    all_scores[i] = p_true
                    
                except Exception as e:
                    print(f"Error parsing API response for prompt {i}: {e}")
                    incomplete_prompts.append(prompts[i])
                    incomplete_texts.append(text)
                    incomplete_indices.append(i)
                    continue
        
        # Handle incomplete responses - call API again with fixed prompts
        if incomplete_indices and not self.use_classifier:
            print("Encountered incomplete responses, fixing with API...")
            self.fix_call_count += 1
            fixed_texts, fixed_counts, fixed_scores = self._fix_incomplete_responses_api(
                incomplete_prompts, incomplete_texts
            )
            for orig_idx, (text, count, score) in zip(
                incomplete_indices, zip(fixed_texts, fixed_counts, fixed_scores)
            ):
                all_outputs[orig_idx] = text
                all_output_token_counts[orig_idx] = count
                all_scores[orig_idx] = score
        
        # Save prompts and outputs to test1.jsonl
        try:
            with open("test2.jsonl", "a", encoding="utf-8") as f:
                for prompt, output, score in zip(prompts, all_outputs, all_scores):
                    log_entry = {
                        "prompt": prompt,
                        "output": output,
                        "score": score
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Warning: Failed to save to test1.jsonl: {e}")
        
        # import pdb; pdb.set_trace()
        return all_outputs, all_output_token_counts, all_scores
    
    def _fix_incomplete_responses_api(
        self, original_prompts: List[str], generated_texts: List[str]
    ) -> Tuple[List[str], List[int], List[float]]:
        """Fix incomplete responses using API."""
        cleaned_texts = []
        for text in generated_texts:
            text = text.rstrip()
            
            if not text.endswith(('.', '!', '?')):  # remove incomplete sentences
                last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last_punct != -1:
                    text = text[: last_punct + 1]
            cleaned_texts.append(text.strip())
        
        forced_prompts = [
            f"{original_prompt}\n{cleaned_text}\n</think>"
            for original_prompt, cleaned_text in zip(original_prompts, cleaned_texts)
        ]
        
        try:
            # Call OpenAI completions API with forced true/false tokens only
            response = self._openai_client.completions.create(
                model="jhu-clsp/rank1-7b",
                prompt=forced_prompts,
                temperature=0,
                max_tokens=1,
                logprobs=20,
            )
            # Convert response to dict for compatibility
            result = response.model_dump()
        except Exception as e:
            print(f"Error calling API for fix: {e}")
            # Return default values
            return (
                [text for text in generated_texts],
                [len(self._tokenizer.encode(text)) for text in generated_texts],
                [0.5] * len(generated_texts)
            )
        
        all_final_texts, all_token_counts, all_scores = [], [], []
        
        if "choices" in result:
            for i, choice in enumerate(result["choices"]):
                try:
                    text = choice.get("text", "")
                    logprobs_data = choice.get("logprobs", {})
                    
                    # Extract final token logprobs
                    final_logprobs_dict = {}
                    if "top_logprobs" in logprobs_data and logprobs_data["top_logprobs"]:
                        final_logprobs_dict = logprobs_data["top_logprobs"][-1]
                    elif "content" in logprobs_data and logprobs_data["content"]:
                        final_token_data = logprobs_data["content"][-1]
                        top_logprobs = final_token_data.get("top_logprobs", [])
                        final_logprobs_dict = {item["token"]: item["logprob"] for item in top_logprobs}
                    
                    true_token_str = self._tokenizer.decode([self.true_token])
                    false_token_str = self._tokenizer.decode([self.false_token])
                    
                    true_logit = None
                    false_logit = None
                    
                    for token_key, logprob in final_logprobs_dict.items():
                        if isinstance(token_key, str):
                            if token_key.strip() == true_token_str.strip():
                                true_logit = logprob
                            elif token_key.strip() == false_token_str.strip():
                                false_logit = logprob
                        elif isinstance(token_key, int):
                            if token_key == self.true_token:
                                true_logit = logprob
                            elif token_key == self.false_token:
                                false_logit = logprob
                    
                    if true_logit is None or false_logit is None:
                        print(f"Warning: Could not find true/false logits in fix response")
                        self.fix_exception_count += 1
                        all_final_texts.append(text)
                        all_token_counts.append(len(self._tokenizer.encode(text)))
                        all_scores.append(0.5)
                        continue
                    
                    true_score = math.exp(true_logit)
                    false_score = math.exp(false_logit)
                    p_true = true_score / (true_score + false_score)
                    
                    all_final_texts.append(text)
                    all_token_counts.append(len(self._tokenizer.encode(text)))
                    all_scores.append(p_true)
                    
                except Exception as e:
                    print(f"Error in fix response parsing: {e}")
                    self.fix_exception_count += 1
                    all_final_texts.append(choice.get("text", ""))
                    all_token_counts.append(len(self._tokenizer.encode(choice.get("text", ""))))
                    all_scores.append(0.5)
        
        return all_final_texts, all_token_counts, all_scores

    _vllm_lock = threading.Lock()

    def run_llm_batched(self, prompts: List[str]):
        # Use API if URL is provided
        if self._api_url is not None:
            return self._call_api_batched(prompts)
        
        # Ensure vLLM engine is not accessed concurrently.
        if self._vllm_batched:
            with self._vllm_lock:
                return self._run_llm_batched_impl(prompts)
        return self._run_llm_batched_impl(prompts)

    def _run_llm_batched_impl(self, prompts: List[str]):
        
        if self._vllm_batched:
            outputs = self._vllm_engine.generate(prompts, self._sampling_params, use_tqdm=False)
            
            # Pre-allocate lists with None values
            total_length = len(prompts)
            all_outputs = [None] * total_length
            all_output_token_counts = [None] * total_length
            all_scores = [None] * total_length

            incomplete_prompts, incomplete_texts, incomplete_indices = [], [], []

            for i, output in enumerate(outputs):
                text = output.outputs[0].text
                # import pdb; pdb.set_trace()
                try:
                    final_logits = output.outputs[0].logprobs[-1] # distribution
                except Exception as e:
                    print(f"Error: {e} on getting final logits: {output.outputs[0]}")
                    incomplete_prompts.append(prompts[i])
                    incomplete_texts.append(text)
                    incomplete_indices.append(i)
                    continue

                if self.true_token not in final_logits or self.false_token not in final_logits:
                    incomplete_prompts.append(prompts[i])
                    incomplete_texts.append(text)
                    incomplete_indices.append(i)
                    continue

                token_count = len(output.outputs[0].token_ids)
                true_logit = final_logits[self.true_token].logprob
                false_logit = final_logits[self.false_token].logprob

                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)
                p_true = true_score / (true_score + false_score)

                all_outputs[i] = text
                all_output_token_counts[i] = token_count
                all_scores[i] = p_true

            # Handle incomplete responses
            if incomplete_indices and not self.use_classifier:
                self.fix_call_count += 1
                fixed_texts, fixed_counts, fixed_scores = self._fix_incomplete_responses(
                    incomplete_prompts, incomplete_texts
                )
                 # Fill in the fixed responses at their original positions
                for orig_idx, (text, count, score) in zip(
                    incomplete_indices, zip(fixed_texts, fixed_counts, fixed_scores)
                ):
                    all_outputs[orig_idx] = text
                    all_output_token_counts[orig_idx] = count
                    all_scores[orig_idx] = score

            return all_outputs, all_output_token_counts, all_scores
        
        else:
            inputs = self._tokenizer(prompts, padding=True, return_tensors="pt").to(self._device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    generation_config=self._generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            total_length = len(prompts)
            all_outputs = [None] * total_length
            all_output_token_counts = [None] * total_length
            all_scores = [None] * total_length

            incomplete_prompts, incomplete_texts, incomplete_indices = [], [], []

            for i, sequence in enumerate(outputs.sequences):
                text = self._tokenizer.decode(sequence, skip_special_tokens=False)
                
                if "</think>" not in text:
                    incomplete_prompts.append(prompts[i])
                    incomplete_texts.append(text)
                    incomplete_indices.append(i)
                    continue

                token_count = len(sequence)
                final_logits = outputs.scores[-1][i]
                true_false_logits = final_logits[[self.true_token, self.false_token]]
                probs = torch.softmax(true_false_logits, dim=-1)
                p_true = probs[0].item()

                all_outputs[i] = text
                all_output_token_counts[i] = token_count
                all_scores[i] = p_true

            if incomplete_indices:
                self.fix_call_count += 1
                fixed_texts, fixed_counts, fixed_scores = self._fix_incomplete_responses(
                incomplete_prompts, incomplete_texts
            )
                for orig_idx, (text, count, score) in zip(
                incomplete_indices, zip(fixed_texts, fixed_counts, fixed_scores)):
                    all_outputs[orig_idx] = text
                    all_output_token_counts[orig_idx] = count
                    all_scores[orig_idx] = score

        return all_outputs, all_output_token_counts, all_scores

    def run_llm(self, prompt: str): # single example without batch inference
        outputs, token_counts, scores = self.run_llm_batched([prompt])
        return outputs[0], token_counts[0], scores[0]

    def score_query(self, query: str, texts: List[str]) -> List[float]:
        """Score a single query against a list of passages."""
        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            prompts = [
                self._truncate_by_context(self.return_prompt(query, d))
                for d in batch_texts
            ]
            _, _, batch_scores = self.run_llm_batched(prompts)
            scores.extend(batch_scores)
        return scores

    def transform(self, df: pd.DataFrame, verbose=True):
        probs = []
        final_texts = []

        queries = df['query'].tolist()
        texts = df[self.text_field].tolist()

        iterator = range(0, len(queries), self.batch_size)

        if verbose:
            iterator = tqdm(iterator, desc="rank1", unit="batch")

        for st in iterator:
            q_batch = queries[st: st + self.batch_size]
            d_batch = texts[st: st + self.batch_size]
            
            prompts = [
                self._truncate_by_context(self.return_prompt(q, d))
                for q, d in zip(q_batch, d_batch)
            ]
            outputs_batch, token_counts_batch, p_batch = self.run_llm_batched(prompts)  # p(true)

            probs.extend(p_batch)
            final_texts.extend(outputs_batch)


        df = df.drop(columns=['score', 'rank'], errors='ignore')
        df = df.assign(score=probs, final_text=final_texts)

        df = add_ranks(df)
        return df

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="jhu-clsp/rank1-7b")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--rerank_output_path", type=str, required=True)
    parser.add_argument("--qrels_output_path", type=str)
    parser.add_argument("--text_output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_size", type=int, default=450)

    args = parser.parse_args()

    ranker = Rank1(
        model=args.model_name_or_path,
        batch_size=args.batch_size,
        context_size=args.context_size,
    )
    

    rows = []
    with open(args.dataset_path, "r", encoding="utf-8") as r:
        for line in r:
            data = json.loads(line.strip())

            rows.append({
                "query_id": data["query_id"],
                "docid": data["docid"],
                "query": data["query"],
                "text": data["text"],
            })

    df = pd.DataFrame(rows)
    df = ranker.transform(df,verbose=True)
"""