from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import jinja2
import requests
from prompts import format_query
from rich import print as rprint
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType


class SearchToolHandler:

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "local_knowledge_base_retrieval":
            return self._search(arguments["user_query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "local_knowledge_base_retrieval",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_query": {
                                "type": "string",
                                "description": "Query to search the local knowledge base for relevant information",
                            }
                        },
                        "required": ["user_query"],
                        "additionalProperties": False,
                    },
                }
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_document",
                        "description": self.searcher.get_document_description(),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "docid": {
                                    "type": "string",
                                    "description": "Document ID to retrieve",
                                }
                            },
                            "required": ["docid"],
                            "additionalProperties": False,
                        },
                    }
                }
            )

        return tools

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


def strftime_now_function(fmt):
    return datetime.now().strftime(fmt)


def get_stream_response(url, headers, payload, print_response=False, timeout=1200, max_retries=3):
    """
    Make API request with timeout and retry logic.
    
    Args:
        url: API endpoint URL
        headers: Request headers
        payload: Request payload
        print_response: Whether to print the response
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Full response text
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff: wait 2^attempt seconds (max 60s)
                wait_time = min(2 ** attempt, 60)
                print(f"Retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            full_response = data['choices'][0]['text']
            
            if print_response:
                print(full_response)
            return full_response
            
        except requests.exceptions.Timeout as e:
            last_exception = e
            print(f"Timeout error on attempt {attempt + 1}/{max_retries}: {str(e)}")
        except requests.exceptions.ConnectionError as e:#
            last_exception = e
            print(f"Connection error on attempt {attempt + 1}/{max_retries}: {str(e)}")
        except requests.exceptions.HTTPError as e:
            last_exception = e
            print(f"HTTP error on attempt {attempt + 1}/{max_retries}: {str(e)}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            print(f"Request error on attempt {attempt + 1}/{max_retries}: {str(e)}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            last_exception = e
            print(f"Response parsing error on attempt {attempt + 1}/{max_retries}: {str(e)}")
        except Exception as e:
            last_exception = e
            print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {str(e)}")
    
    # If all retries failed, raise the last exception
    print(f"All {max_retries} attempts failed. Raising last exception.")
    raise last_exception

def validate_response(reasoning_content, content):
    # No unwanted reasoning special tokens
    unwanted_reasoning_special_tokens = ["seed:cot_budget_reflect", "seed:tool_call", "seed:eos"]
    if any(token in reasoning_content for token in unwanted_reasoning_special_tokens):
        return False
    # No unwanted content special tokens
    unwanted_content_special_tokens = ["seed:cot_budget_reflect", "seed:think"]
    if any(token in content for token in unwanted_content_special_tokens):
        return False
    # Both starting and ending tool call tags must be present
    starting_tool_call_in_content = "<seed:tool_call>" in content
    ending_tool_call_in_content = "</seed:tool_call>" in content
    if starting_tool_call_in_content ^ ending_tool_call_in_content:
        return False
    # Tool call in right format
    if (not starting_tool_call_in_content) and (not ending_tool_call_in_content):
        function_call_keywords = ["parameter", "user_query", "local_knowledge_base_retrieval"]
        if all(keyword in content for keyword in function_call_keywords):
            return False
    # Reasoning has no tool call wit
    return True

def execute_tool_from_response(llm_generated_response, tool_handler):
    """
    Parse and execute tool call from LLM response text.
    
    Args:
        llm_generated_response: The LLM response text containing tool call in format:
            <seed:tool_call>
            <function=tool_name>
            <parameter=param_name>param_value</parameter>
            </function>
            </seed:tool_call><seed:eos>
        tool_handler: SearchToolHandler instance
        
    Returns:
        Tuple of (result, tool_name, tool_args)
    """
    try:
        tool_call_section = llm_generated_response.split('<seed:tool_call>')[-1].split('</seed:tool_call>')[0]
        
        # Parse tool name from <function=tool_name>
        tool_name = tool_call_section.split('<function=')[-1].split('>')[0].strip()
        
        # Parse parameters
        tool_args = {}
        # Find all parameter tags
        import re
        param_pattern = r'<parameter=([^>]+)>([^<]*)</parameter>'
        matches = re.findall(param_pattern, tool_call_section)
        
        for param_name, param_value in matches:
            tool_args[param_name] = param_value
        
        # Execute tool
        result = tool_handler.execute_tool(tool_name, tool_args)
        
    except Exception as e:
        print(f'Error executing tool call: {e}, full response: {llm_generated_response}')
        result = f'Error: Tool call is not valid: {str(e)}'
        return result, None, None
    
    return result, tool_name, tool_args


def run_conversation_with_tools(
    model_url: str,
    template,
    initial_request: dict,
    tool_handler: SearchToolHandler,
    max_iterations: int = 100,
    verbose: bool = False,
):
    """
    Run conversation with tools using HTTP requests and Jinja template.
    
    Args:
        model_url: Base URL for the model API
        template: Jinja2 template for rendering prompts
        initial_request: Initial request dict containing model, max_tokens, etc.
        tool_handler: SearchToolHandler instance
        max_iterations: Maximum number of iterations
        verbose: Whether to print verbose logs
        
    Returns:
        Tuple of (messages, tool_usage, status)
    """
    tool_usage = {}
    
    # Convert tools to the format expected by template
    tools = tool_handler.get_tool_definitions()
    
    messages = initial_request["input"]
    
    # Render initial prompt
    prompt = template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True
    )
    
    # Prepare API endpoint
    url = f"{model_url}/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    
    # Track conversation for persistence
    conversation_messages = []
    conversation_messages.extend(messages)
    
    iteration = 1
    
    while iteration <= max_iterations:
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": initial_request.get("max_output_tokens", 10000),
                "stream": False,
                "skip_special_tokens": False,
                # "stop": ["<|call|>"],
                "include_stop_str_in_output": True,
            }
            
            response_text = get_stream_response(url, headers, payload, print_response=verbose)
            reasoning = response_text.split("<seed:think>")[-1].split("</seed:think>")[0]
            content = response_text.split("</seed:think>")[-1]

            if not validate_response(reasoning, content):
                print(f"Skipping invalid response: {content}")
                print("================================================")
                iteration += 1
                continue
            
            if "<seed:tool_call>" in content:
                # print(f"Tool call found in content: {content}")
                # print("================================================")
                result, tool_name, tool_args = execute_tool_from_response(content, tool_handler)
                conversation_messages.append({
                    "role": "assistant",
                    "reasoning_content": reasoning,
                    "tool_calls": [
                        {
                            "name": tool_name,
                            "arguments": tool_args,
                        }
                    ]
                })
                conversation_messages.append({
                    "role": "tool",
                    "content": result,
                })
                prompt = template.render(
                    messages=conversation_messages,
                    tools=tool_handler.get_tool_definitions(),
                    add_generation_prompt=True
                )
            else:
                conversation_messages.append({
                    "role": "assistant",
                    "reasoning_content": reasoning,
                    "content": content,
                })
                print(f"Completed response in iteration {iteration}")
                return conversation_messages, tool_usage, "completed"
                
        except Exception as e:
            conversation_messages.append({
                "type": "error",
                "content": [{"type": "error_text", "text": str(e)}],
            })
            return conversation_messages, tool_usage, "error"
        
        iteration += 1
    
    return conversation_messages, tool_usage, "incomplete"


def _persist_response(
    out_dir: str,
    initial_request: dict,
    messages: list,
    tool_usage: dict,
    status: str,
    *,
    query_id: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    call_output_by_id: dict[str, str | dict | None] = {}
    for item in messages or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            call_output_by_id[item.get("call_id")] = item.get("output")

    normalized_results: list[dict] = []
    
    for i, item in enumerate(messages or []):
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        itype = item.get("type")

        # Handle new format from run_conversation_with_tools (role-based)
        if role == "assistant":
            # Extract reasoning content if present
            reasoning_content = item.get("reasoning_content")
            if reasoning_content:
                reasoning_output = [reasoning_content] if isinstance(reasoning_content, str) else reasoning_content
                normalized_results.append(
                    {
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": reasoning_output,
                    }
                )
            
            # Handle tool calls
            tool_calls = item.get("tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})
                    
                    # Look ahead for corresponding tool output
                    tool_output = None
                    if i + 1 < len(messages):
                        next_item = messages[i + 1]
                        if isinstance(next_item, dict) and next_item.get("role") == "tool":
                            tool_output = next_item.get("content")
                            # Parse JSON if it's a string
                            if isinstance(tool_output, str):
                                try:
                                    tool_output = json.loads(tool_output)
                                except (json.JSONDecodeError, ValueError):
                                    pass
                    
                    normalized_results.append(
                        {
                            "type": "tool_call",
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "output": tool_output,
                        }
                    )
                    
                    # Track tool usage
                    if tool_name:
                        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
            
            # Handle final content (no tool calls)
            content = item.get("content")
            if content and not tool_calls:
                text = content if isinstance(content, str) else str(content)
                if text.strip():
                    normalized_results.append(
                        {
                            "type": "output_text",
                            "tool_name": None,
                            "arguments": None,
                            "output": text,
                        }
                    )
        
        # Handle tool role messages (already processed above when looking ahead)
        elif role == "tool":
            # Skip - already handled when processing assistant messages with tool_calls
            continue
        
        # Handle user role messages (skip - not needed in normalized results)
        elif role == "user":
            continue
        
        # Handle old format (type-based) for backward compatibility
        elif itype == "function_call":
            call_id = item.get("call_id")
            normalized_results.append(
                {
                    "type": "tool_call",
                    "tool_name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "output": call_output_by_id.get(call_id),
                }
            )

        elif itype == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list) and len(summary) > 0:
                reasoning_output = summary
            else:
                reasoning_output = []
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") in {
                        "reasoning_text",
                        "output_text",
                        "text",
                    }:
                        text_val = str(part.get("text", "")).strip()
                        if text_val:
                            reasoning_output.append(text_val)
            normalized_results.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": reasoning_output,
                }
            )

        elif itype == "message":
            parts = item.get("content", []) or []
            text_chunks: list[str] = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_chunks.append(str(part.get("text", "")))
            text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            if text:
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": text,
                    }
                )

        elif itype == "error":
            normalized_results.append(
                {
                    "type": "error",
                    "content": item.get("content"),
                }
            )

    normalized_tool_counts: dict[str, int] = {}
    for tool_name, count in (tool_usage or {}).items():
        normalized_name = (
            "search"
            if isinstance(tool_name, str) and ("retrieval" in tool_name.lower())
            else tool_name
        )
        normalized_tool_counts[normalized_name] = normalized_tool_counts.get(
            normalized_name, 0
        ) + int(count or 0)

    normalized_record = {
        "metadata": {
            "model": initial_request.get("model"),
            "reasoning": initial_request.get("reasoning"),
            "output_dir": str(out_dir),
        },
        "query_id": query_id,
        "tool_call_counts": normalized_tool_counts,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(normalized_record, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", normalized_tool_counts)


def _process_tsv_dataset(
    tsv_path: str, model_url: str, template, args, tool_handler: SearchToolHandler
):
    """Process a TSV file of (id \t query) pairs sequentially and persist responses."""

    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue  # ignore corrupt files

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    import threading

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        """Build request, send and persist response for one query."""

        initial_request = {
            "model": args.model,
            "max_output_tokens": args.max_tokens,
            "input": [
                {"role": "user", "content": format_query(qtext, args.query_template)}
            ],
            "tools": tool_handler.get_tool_definitions(),
            "truncation": "auto",
            "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
        }

        try:
            messages, tool_usage, status = run_conversation_with_tools(
                model_url, template, initial_request, tool_handler, args.max_iterations, args.verbose
            )

            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                out_dir, initial_request, messages, tool_usage, status, query_id=qid
            )

        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = [
                executor.submit(_handle_single_query, qid, qtext, pbar)
                for qid, qtext in remaining
            ]

            for _ in as_completed(futures):
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Call vLLM OpenAI Completions API with Jinja template and local search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="User query *or* TSV file path",
    )
    parser.add_argument(
        "--model", default="openai/gpt-oss-20b", help="Model name served by vLLM"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="max_tokens for Completions API",
    )
    parser.add_argument(
        "--output-dir", default="runs_vllm", help="Directory to store run JSON files"
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use (default: %(default)s)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=1, help="Parallel threads for dataset mode"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max conversation rounds with function calls",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "--model-url", default="http://localhost:8000/v1", help="Model URL"
    )
    parser.add_argument(
        "--template-path",
        default="./seed_oss_chat_template.jinja",
        help="Path to Jinja template file",
    )

    # Searcher selection and shared tool options --------------------------
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models",
    )
    parser.add_argument(
        "--hf-home", type=str, help="Hugging Face home directory for caching"
    )

    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Tokens per document snippet",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Top-k search results to return for each query"
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="Also register the get_document tool",
    )

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    # Load Jinja template
    template_path = Path(args.template_path)
    if not template_path.is_file():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_string = f.read()
    
    env = jinja2.Environment()
    env.globals['strftime_now'] = strftime_now_function
    env.filters['tojson'] = json.dumps
    template = env.from_string(template_string)

    searcher = searcher_class(args)

    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    print("Processing TSV dataset", potential_path)
                    _process_tsv_dataset(
                        str(potential_path), args.model_url, template, args, tool_handler
                    )
                    return
            except OSError:
                pass

    print("Processing single query", args.query)

    args.query = format_query(args.query, args.query_template)

    messages = [{"role": "user", "content": args.query}]

    initial_request = {
        "model": args.model,
        "max_output_tokens": args.max_tokens,
        "input": messages,
        "tools": tool_handler.get_tool_definitions(),
        "truncation": "auto",
        "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
    }

    messages, tool_usage, status = run_conversation_with_tools(
        args.model_url, template, initial_request, tool_handler, args.max_iterations, args.verbose
    )

    _persist_response(
        args.output_dir, initial_request, messages, tool_usage, status, query_id=None
    )

    rprint(messages)


if __name__ == "__main__":
    main()

