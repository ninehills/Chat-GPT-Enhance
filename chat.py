import io
import os
import ssl
from contextlib import closing
from typing import Optional, Tuple, List, Any, Dict
import datetime

import gradio as gr
import requests
import inscriptis

from langchain import ConversationChain, LLMChain

from langchain.agents import initialize_agent, Tool, load_tools
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from threading import Lock

# Console to variable
from io import StringIO
import sys
import re

from openai.error import AuthenticationError, InvalidRequestError, RateLimitError

# Pertains to Express-inator functionality
from langchain.prompts import PromptTemplate

# Pertains to question answering functionality
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain.utilities import ApifyWrapper
from langchain.tools import BaseTool

MAX_TOKENS = 512

class WebCrawlerTool(BaseTool):
    name = "web_crawler"
    description = "A portal to the internet. Use this when you need to get specific content from a website. Input should be a  url (i.e. https://www.google.com). The output will be the text response of the GET request."

    headers: Optional[Dict[str, str]] = None

    def _run(self, url: str) -> str:
        """Use the tool."""
        html = self._get(url, timeout=10)
        return inscriptis.get_text(html)

    async def _arun(self, url: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("WebCrawlerTool does not support async")

    def _get(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text."""
        return requests.get(url, headers=self.headers, **kwargs).text


def update_settings(use_gpt4, tools, openai_api_key, serper_api_key, wolfram_alpha_appid):
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
    if wolfram_alpha_appid:
        os.environ["WOLFRAM_ALPHA_APPID"] = wolfram_alpha_appid
    if use_gpt4:
        llm = ChatOpenAI(temperature=0, max_tokens=MAX_TOKENS, model_name="gpt-4")
        print("Trying to use llm OpenAIChat with gpt-4")
    else:
        print("Trying to use llm OpenAI with gpt-3.5-turbo")
        llm = ChatOpenAI(temperature=0, max_tokens=MAX_TOKENS, model_name="gpt-3.5-turbo")
    
    tools_list = []
    for tool in tools:
        print("Load tool: ", tool)
        if tool == "google-serper'":
            t = Tool(
                name="Serper Search",
                func=GoogleSerperAPIWrapper(serper_api_key=serper_api_key).run,
                description="A low-cost Google Search API. Useful for when you need to answer questions about current events. Input should be a search query.",
            )
        elif tool == "wolfram-alpha":
            t = WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper(wolfram_alpha_appid=wolfram_alpha_appid))
        elif tool == "web_crawler":
            t = WebCrawlerTool()
        else:
            raise ValueError("Tool not found: " + tool)
        tools_list.append(t)

    # memory = ConversationBufferMemory(memory_key="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = initialize_agent(tools_list, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
    express_chain = LLMChain(llm=llm, prompt=PROMPT_TEMPLATE, verbose=True)
    return chain, express_chain, memory


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(
            self, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain],
            trace_chain: bool, monologue: bool, express_chain: Optional[LLMChain],
            translate_to: str, force_translate: bool
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
            print("inp: " + inp)
            print("trace_chain: ", trace_chain)
            print("monologue: ", monologue)
            history = history or []
            # If chain is None, that is because no API key was provided.
            output = "发生错误，请检查 API Key 是否正确"
            hidden_text = output

            if chain:
                if not monologue:
                    output, hidden_text = run_chain(chain, inp, capture_hidden_text=trace_chain)
                else:
                    output, hidden_text = inp, None
            else:
                print("chain is not found.")

            output = transform_text(output, express_chain, translate_to, force_translate)

            text_to_display = output
            if trace_chain:
                text_to_display = hidden_text + "\n\n" + output
            history.append((inp, text_to_display))

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, ""


def run_chain(chain: LLMChain, inp: str, capture_hidden_text: bool):
    output = ""
    hidden_text = None
    if capture_hidden_text:
        error_msg = None
        tmp = sys.stdout
        hidden_text_io = StringIO()
        sys.stdout = hidden_text_io

        try:
            output = chain.run(input=inp)
        except AuthenticationError as ae:
            error_msg = str(ae) + str(datetime.datetime.now()) + ". " + str(ae)
            print("error_msg", error_msg)
        except RateLimitError as rle:
            error_msg = "\n\nRateLimitError: " + str(rle)
        except ValueError as ve:
            import traceback
            traceback.print_exc()
            error_msg = "\n\nValueError: " + str(ve)
        except InvalidRequestError as ire:
            error_msg = "\n\nInvalidRequestError: " + str(ire)
        except Exception as e:
            error_msg = "\n\n" + "Exception" + ":\n\n" + str(e)

        sys.stdout = tmp
        hidden_text = hidden_text_io.getvalue()

        # remove escape characters from hidden_text
        hidden_text = re.sub(r'\x1b[^m]*m', '', hidden_text)

        # remove "Entering new AgentExecutor chain..." from hidden_text
        hidden_text = re.sub(r"Entering new AgentExecutor chain...\n", "", hidden_text)

        # remove "Finished chain." from hidden_text
        hidden_text = re.sub(r"Finished chain.", "", hidden_text)

        # Add newline after "Thought:" "Action:" "Observation:" "Input:" and "AI:"
        hidden_text = re.sub(r"Thought:", "\n\nThought:", hidden_text)
        hidden_text = re.sub(r"Action:", "\n\nAction:", hidden_text)
        hidden_text = re.sub(r"Observation:", "\n\nObservation:", hidden_text)
        hidden_text = re.sub(r"Input:", "\n\nInput:", hidden_text)
        hidden_text = re.sub(r"AI:", "\n\nAI:", hidden_text)

        if error_msg:
            hidden_text += error_msg

        print("hidden_text: ", hidden_text)
    else:
        try:
            output = chain.run(input=inp)
        except AuthenticationError as ae:
            output = str(ae) + str(datetime.datetime.now()) + ". " + str(ae)
            print("output", output)
        except RateLimitError as rle:
            output = "\n\nRateLimitError: " + str(rle)
        except ValueError as ve:
            output = "\n\nValueError: " + str(ve)
        except InvalidRequestError as ire:
            output = "\n\nInvalidRequestError: " + str(ire)
        except Exception as e:
            output = "\n\n" + "Exception" + ":\n\n" + str(e)

    return output, hidden_text


TRANSLATE_TO_DEFAULT = "N/A"
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["original_words", "translate_to"],
    template="Restate {translate_to}the following: \n{original_words}\n",
)

# Pertains to Express-inator functionality
def transform_text(desc: str, express_chain: LLMChain, translate_to: str, force_translate: bool):
    translate_to_str = ""
    if translate_to != TRANSLATE_TO_DEFAULT and force_translate:
        translate_to_str = "translated to " + translate_to + ", "


    formatted_prompt = PROMPT_TEMPLATE.format(
        original_words=desc,
        translate_to=translate_to_str,
    )

    trans_instr = translate_to_str
    if express_chain and len(trans_instr.strip()) > 0:
        generated_text = express_chain.run(
            {'original_words': desc, 'translate_to': translate_to_str}).strip()
    else:
        print("Not transforming text")
        generated_text = desc

    # replace all newlines with <br> in generated_text
    generated_text = generated_text.replace("\n", "\n\n")

    prompt_plus_generated = "GPT prompt: " + formatted_prompt + "\n\n" + generated_text

    print("\n==== date/time: " + str(datetime.datetime.now() - datetime.timedelta(hours=5)) + " ====")
    print("prompt_plus_generated: " + prompt_plus_generated)

    return generated_text