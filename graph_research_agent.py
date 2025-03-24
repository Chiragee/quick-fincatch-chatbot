import os
import time
import copy

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError

import tiktoken
ENCODING = tiktoken.encoding_for_model('gpt-4o')

def get_tiktoken_token_count(content):
    return len(ENCODING.encode(content))

def call_gemini_complete(prompt = None, model_name = 'gemini-2.0-flash', tools = None, tool_config = None, max_retries=3):
    #have to selectively choose which model to use.

    # model_name = 'gemini-2.0-flash-thinking-exp'
    # model_name = 'gemini-2.0-flash'

    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    model = genai.GenerativeModel(model_name)

    messages = [{"role": "user", "parts": [{'text': prompt}]}]
    
    current_iteration = 0
    while current_iteration < max_retries + 1:

        try:

            response = model.generate_content(contents = messages, tools=tools, tool_config=tool_config)  # Removed kwargs because Google cannot handle kwargs that aren't applicable

            if response.parts:

                all_text = []
                function_call = None
                for part in response.parts:
                    if text := part.text:
                        all_text.append(text)
                    elif fn := part.function_call:
                        function_call = fn
                    else:
                        print(f"Unexpected part: {part}")
                        continue
                
                to_return = []
                if all_text:
                    to_return.append(
                        {
                            "type": "text",
                            "text": os.linesep.join(all_text)
                        }
                    )
                if function_call:
                    to_return.append(
                        {
                            "type": "function_call", 
                            "name": function_call.name, 
                            "arguments": function_call.args
                        }
                    )

                return to_return

        except (ResourceExhausted, InternalServerError, ValueError) as e:
            if current_iteration >= max_retries:
                print(f"Failed to get response from model after current iteration: {current_iteration}")
                raise e
            print(f"Retryable error occured: {str(e)}, waiting 5 seconds and retrying.")
            time.sleep(5)
            current_iteration+=1
            continue
        except Exception as e:
            print(f"Unexpected exception: {e}")
            raise e
    
    raise Exception(f"Failed to get response from model after current iteration: {current_iteration}")
    

system_instructions  = '''You are a graph research assistant that can work step by step, call functions to gather information, and ultimately aims to achieve the users objective. 

**INSTRUCTIONS:**
- At each step your response and function responses will be fed back to you in order for you to proceed. The idea is for you to call the query_graph function, get the results, determine if you need to gather more information, call it again, aggregate the information, and when you are finally done, call the finish_response function to provide your final answer.
- You are also given access to a persistent 'notepad' where you should write down important findings, plan, questions you have, or anything else you think is important. The notepad is persistent and will be available throughout, the history will go away, but the notepad content will not, and you can update it accordingly using the write_to_notepad function. Suggested to update every 5 iterations.
- Your history, your responses, and function responses will be repeatedly passed back to you in order for you to proceed.
- You do NOT have to call a function on every single iteration, it is completely okay for you to spend some time to think about the information you have gathered so far and decide if you need more information or if you are ready to finish. In this case, you responses will simply be fed back to you in order for you to proceed.
- You have a maximum number of iterations in order to achieve the user's objective. You are currently on iteration {current_iteration} out of a maximum of {max_iterations}. You MUST call the finish_response function before the maximum number of iterations is reached.
- When you are finished with your research, you must call the finish_response function to provide your final answer. An explicit function call to the finish_response function is required to end the loop.
- You must be factual, thorough, and aggregate information from the calls.
- Citations:
    - Ensure that you cite the information id's using square brackets in your response. For example: "this is some information [1234]". This is essential for the user to be able to verify the information.
    - The user does not have access to the content retrieved from the graph database, so you must provide all relevant information in your response. i.e dont say according to [1234] the answer is X. You must actually provide the specific answer in full.
    - Absolutely under no circumstances, should you make up any citations or provide false information. You must only provide information based on analysis of results that you have gathered from the functions you call.
    - For citations already passed in the objective, you must propogate them to the final response where appropriate, and do not modify these citations. PROPAGATE THE ORIGINAL CITATIONS.
- You must return a 'content' string in your final response.
    - 'content': This should be your final answer to the users objective in markdown structured format using all of the analysis you have conducted. Your output format must be purely structured markdown. No preliminary comments or markdown tags are allowed, your response must directly answer the users query and be in markdown format. Unless specified otherwise in the users objective, you must use inplace square bracket citations in your response. For example, "The sky is blue [1][2][3]."
- Output format on every iteration:
    - You must always output structured markdown for your thoughts, analysis and results, (but not function calls). No markdown prefixes or suffixes are allowed (i.e not allowed a starting ```markdown or ending ```).
- You must aggregate citations from the responses you get from the functions you call, and make sure that the final response includes all the citations appropriately.
- Use your best judgement everywhere applicable.

**CURRENT DATE:**
{current_date}

**YOUR HISTORY:**
{history}

**USER OBJECTIVE:**
{objective}

**NOTEPAD:**
{notepad}

**FUNCTIONS:**
You have access to 3 functions:
1. query_graph
- This function takes a query string as input, and queries our global financial knowledge graph to retrieve context that is meant to be relevant to the query. 

2. write_to_notepad
- This function takes a content string as input, and adds it to your notepad. It will return a success message.

3. finish_response
- This function takes a content string and a citations array as input and doesnt return anything. Use this function to return your final results to the user, the loop will end after this.

You may call 1 function at a time.'''

forced_finish_instructions  = '''You are a research assistant that can work step by step, call functions to gather information, and ultimately aims to achieve the users objective.

**YOU HAVE REACHED THE MAXIMUM NUMBER OF ITERATIONS AND MUST RETURN A 'finish_response' FUNCTION CALL NOW.**

- Citations:
    - Ensure that you cite the information id's using square brackets in your response. For example: "this is some information [1234]". This is essential for the user to be able to verify the information.
    - The user does not have access to the content retrieved from the graph database, so you must provide all relevant information in your response. i.e dont say according to [1234] the answer is X. You must actually provide the specific answer in full.
    - Absolutely under no circumstances, should you make up any citations or provide false information. You must only provide information based on analysis of results that you have gathered from the functions you call.
    - For citations already passed in the objective, you must propogate them to the final response where appropriate, and do not modify these citations. PROPAGATE THE ORIGINAL CITATIONS.
- You must return a 'content' string in your final response.
    - 'content': This should be your final answer to the users objective in markdown structured format using all of the analysis you have conducted. Your output format must be purely structured markdown. No preliminary comments or markdown tags are allowed, your response must directly answer the users query and be in markdown format. Unless specified otherwise in the users objective, you must use inplace square bracket citations in your response. For example, "The sky is blue [1][2][3]."

**INSTRUCTIONS:**
- You must call the finish_response function to provide your final answer and citations based on the history and the users objective.

**CURRENT DATE:**
{current_date}

**YOUR HISTORY:**
{history}

**USER OBJECTIVE:**
{objective}

**NOTEPAD:**
{notepad}

**FUNCTIONS:**
You have access to ONE function which you MUST CALL NOW:

1. finish_response
- This function takes a content string and a citations array as input and doesnt return anything. Use this function to return your final results to the user, the loop will end after this.
'''


################################### TOOLS ###################################
from functions import get_context
def query_graph(query, start_date = '2024-06-01', end_date = '2025-03-22', context_window = 10000, k = 50):
    return get_context(query, start_date=start_date, end_date=end_date, context_window=context_window, k=k)
query_graph_schema = {
    "name": "query_graph",
    "description": "This function takes a query string as input, queries the global financial knowledge graph to retrieve context relevant to the query, and returns the results. Will return 10000 tokens of context at a time, so ensure that you understand how to use this query effectively.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query string to search in the global financial knowledge graph. Results are retrieved based on cosine similarity + personalised pagerank compared for the input query. Ensure that you understand how to use this query effectively."
            },
            "start_date": {
                "type": "string",
                "description": "The start date for the context window. Default is '2024-06-01'. Use this if you desire to have more targeted queries."
            },
            "end_date": {
                "type": "string",
                "description": "The end date for the context window. Default is '2025-03-22'. This cannot be a future date. Use this if you desire to have more targeted queries."
            },
            # "context_window": {
            #     "type": "integer",
            #     "description": "The size of the context window. Default is 10000."
            # },
            # "k": {
            #     "type": "integer",
            #     "description": "The number of results to return. Default is 50."
            # }
        },
        "required": ["query"]
    }

}

write_to_notepad_schema = {
    "name": "write_to_notepad",
    "description": "This function takes a content string as input and adds it to your notepad. It returns a success message.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to be added to the notepad."
            }
        },
        "required": ["content"]
    }
}

finish_response_schema = {
    "name": "finish_response",
    "description": "Returns final results to the user.",
    "parameters": {
        "type": "object",
        "properties": {
        "content": {
            "type": "string",
            "description": "Final response content for the user."
        },
        },
        "required": ["content"]
    }
}

######################################################################

def graph_research_agent(objective, max_iterations = 10):
    #no longer allowed to set model, we just use gemini flash 2 thinking for all for now.
    #model is between 'gemini-exp-1206' and 'gemini-2.0-flash-exp'
    # model_name = 'gemini-2.0-flash-thinking-exp'
    # model = genai.GenerativeModel(model)


    history = []
    notepad = ""
    current_iteration = 0
    proper_finish = False

    content = None
    # citations = None

    while current_iteration <= max_iterations and not proper_finish:
        current_iteration += 1

        print(f"----------------current_iteration: {current_iteration}----------------")

        # BASIC CONTEXT MANAGEMENT
        history_for_llm = copy.copy(history)
        # Format prompt
        prompt = system_instructions.format(history=history_for_llm, objective=objective, notepad=notepad, current_iteration=current_iteration, max_iterations=max_iterations, current_date = time.strftime("%Y-%m-%d"))
        #manage context:
        while get_tiktoken_token_count(prompt) > 50000 and len(history_for_llm) > 3:
            #the only thing we can do is remove from history untill we are under token limit, but we have to leave 3 iterations in, that is the minimum
            history_for_llm = history_for_llm[1:]
            prompt = system_instructions.format(history=history_for_llm, objective=objective, notepad=notepad, current_iteration=current_iteration, max_iterations=max_iterations, current_date = time.strftime("%Y-%m-%d"))
        

        tools = [
            {
                'function_declarations': [query_graph_schema, write_to_notepad_schema, finish_response_schema]
            }
        ]

        # Get response from model
        returned = call_gemini_complete(prompt, tools=tools) #this is a dictionary
        print(f"recieved llm response: {returned}")

        # Add to history
        history.extend(returned)

        #no longer need to parse becayse already a dictionary
        # #parse the response
        # parsed = json_repair.loads(raw_text)

        # Check if we have a function call
        try:

            for parsed in returned:
        
                if parsed.get("type") == "function_call":

                    fn_name = parsed['name']
                    fn_args = parsed['arguments']

                    if fn_name == "query_graph":
                        query = fn_args["query"]
                        start_date = fn_args.get("start_date", '2024-06-01')
                        end_date = fn_args.get("end_date", '2025-03-22')
                        print(f"recieved function call for query_graph with query: {query}, start_date: {start_date}, end_date: {end_date}")

                        # Call the perplexity function
                        result = query_graph(query, start_date=start_date, end_date=end_date)
                        print(f"recieved function response for query_graph: {result[:100]}...")
                        # Add result to history
                        history.append(result)

                        continue

                    elif fn_name == "write_to_notepad":
                        content = fn_args['content']
                        notepad += os.linesep*2 + content
                        print(f"recieved function call for write_to_notepad with content: {content}")

                        # Add to history
                        history.append(f"Added to notepad: {content}")

                        continue


                    elif fn_name == "finish_response":
                        # The agent is done
                        content = fn_args['content']
                        if not content:
                            print(f"finish_response was called but no valid content was provided, you MUST provide a valid content string. got: {content}")
                            history.append(f"finish_response was called but no valid content was provided, you MUST provide a valid content string. got {content}")
                            continue

                        proper_finish = True
                        break

                    else: # Unknown function
                        print(f"Unknown function name: {fn_name} was called, please call a valid function.")
                        history.append(f"Unknown function name: {fn_name} was called, please call a valid function.")

                        continue

                elif parsed.get("type") == "text":
                    # No function call, just reasoning text. we already added it to history. Continue
                    continue

                else:
                    print(f"Invalid response from model: {parsed}")
                    history.append(f"Invalid response from model: {parsed}")
                    continue

            if proper_finish:
                break

        except Exception as e:
            print(f"An error occured: {str(e)}")
            history.append(f"An error occured: {str(e)}")
            continue

    if not proper_finish:

        max_forced_iterations = 3
        current_forced_iteration = 1
        while not proper_finish and current_forced_iteration < max_forced_iterations:
            print(f"----------------current forced iteration: {current_forced_iteration}----------------")
            try:
                # The agent did not call finish_response before max iterations
                print("Agent did not call finish_response before max iterations.")

                history_for_llm = copy.copy(history)
                prompt = forced_finish_instructions.format(history=history_for_llm, objective=objective, notepad=notepad, current_date = time.strftime("%Y-%m-%d"))
                #manage context:
                while get_tiktoken_token_count(prompt) > 50000 and len(history_for_llm) > 3:
                    #the only thing we can do is remove from history untill we are under token limit, but we have to leave 3 iterations in, that is the minimum
                    history_for_llm = history_for_llm[1:]
                    prompt = forced_finish_instructions.format(history=history_for_llm, objective=objective, notepad=notepad, current_date = time.strftime("%Y-%m-%d"))
                
                tools = [
                    {
                        'function_declarations': [finish_response_schema]
                    }
                ] 

                tool_config = {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": ["finish_response"]
                    },
                }


                if current_forced_iteration == max_forced_iterations - 1:
                    print(f"final forced iteration: {current_forced_iteration}, last ditch effort so switching to gemini-1.5-pro-latest")
                    model_name = 'gemini-1.5-pro-latest'
                    returned = call_gemini_complete(prompt, model_name = model_name, tools=tools, tool_config=tool_config)
                else:
                    returned = call_gemini_complete(prompt, tools=tools, tool_config=tool_config)

                # Add to history
                history.extend(returned)

                #no longer need to parse becayse already a dictionary            
                # #parse the response
                # parsed = json_repair.loads(raw_text)
                parsed = returned[0]

                fn_name = parsed['name']
                fn_args = parsed['arguments']

                content = fn_args['content']
                if not content:
                    current_forced_iteration += 1
                    print(f"finish_response was called but no content was provided, you MUST provide a valid content string. got: {content}")
                    history.append(f"finish_response was called but no content was provided, you MUST provide a valid content string. got: {content}")
                    continue

                proper_finish = True

                current_forced_iteration += 1

                break
                
            except Exception as e:
                print(f'''An error occured, YOU MAY ONY CALL THE finish_response function exclusively in the format explained do it NOW: {str(e)}''')
                history.append(f'''An error occured, YOU MAY ONY CALL THE finish_response function exclusively in the format explained do it NOW: {str(e)}''')
                current_forced_iteration += 1
                continue
    
    print(f"returning content: {content}")
    return content


