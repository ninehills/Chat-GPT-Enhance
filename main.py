import gradio as gr

import chat

# Default configs
TOOLS_LIST = ['google-serper', 'wolfram-alpha', 'web_crawler']
TOOLS_DEFAULT_LIST = ['web_crawler']
FORCE_TRANSLATE_DEFAULT = True
USE_GPT4_DEFAULT = False
TRANSLATE_TO_DEFAULT = chat.TRANSLATE_TO_DEFAULT


def reset_memory(history, memory):
    memory.clear()
    history = []
    return history, history, memory

chat_wrapper = chat.ChatWrapper()

with gr.Blocks(title="Chat-GPT-Enhance", theme=gr.themes.Soft()) as block:
    history_state = gr.State()
    memory_state = gr.State()
    chain_state = gr.State()
    express_chain_state = gr.State()
    

    with gr.Row():
        # chat
        with gr.Column(scale=5):
            gr.HTML("""<b><center>Chat-GPT-Enhance</center></b>""")
            chatbot = gr.Chatbot()
            message = gr.Textbox(label="What's on your mind??",
                                 placeholder="What's the answer to life, the universe, and everything?",
                                 lines=1)
            submit = gr.Button(value="Send", variant="secondary").style(full_width=False)
            reset = gr.Button(value="Reset chat", variant="secondary").style(full_width=False)
            reset.click(reset_memory, inputs=[history_state, memory_state],
                            outputs=[chatbot, history_state, memory_state])

            gr.Examples(
                examples=["How many people live in Canada?",
                        "What is 2 to the 30th power?",
                        "If x+y=10 and x-y=4, what are x and y?",
                        "中国在2022年有多少人口？",
                        "Get me information about the movie 'Avatar'",
                        "总结下这个页面：https://iliana.fyi/blog/acropalypse-now/",
                        "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses - "
                        "if I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"],
                inputs=message
            )
        
        # config
        with gr.Column(scale=2):
            gr.HTML("""<b><center>Settings</center></b>""")
            apply = gr.Button(value="Apply Settings", variant="secondary").style(full_width=False)
            gr.HTML("""<b>Below setting need manual apply:</b>""")
            tools_group = gr.CheckboxGroup(
                label="Tools:", choices=TOOLS_LIST, value=TOOLS_DEFAULT_LIST, interactive=True)
            openai_api_key = gr.Textbox(
                label="OpenAI API Key:", placeholder="Paste your OpenAI API key (sk-...)", lines=1, type='password')
            serper_api_key = gr.Textbox(
                label="Serper.dev API Key:", placeholder="serper.dev API KEY", lines=1, type='password')
            wolfram_alpha_appid = gr.Textbox(
                label="Wolfram Alpha APP ID:", placeholder="Wolfram Alpha APP ID", lines=1, type='password')
            use_gpt4 = gr.Checkbox(label="Use GPT-4 (experimental) if your OpenAI API has access to it",
                        value=USE_GPT4_DEFAULT)
            gr.HTML("""<b>Other settings:</b>""")
            trace_chain = gr.Checkbox(label="Show reasoning chain in chat bubble", value=False)
            force_translate = gr.Checkbox(label="Force translation to selected Output Language",
                                            value=FORCE_TRANSLATE_DEFAULT)
            monologue = gr.Checkbox(label="Babel fish mode (translate/restate what you enter, no conversational agent)",
                                    value=False)
            
            gr.HTML("""<b><center>Language</center></b>""")
            translate_to = gr.Radio(label="Language:", choices=[
                TRANSLATE_TO_DEFAULT, "Chinese", "English"], value=TRANSLATE_TO_DEFAULT)
            
            apply.click(chat.update_settings, inputs=[use_gpt4, tools_group, openai_api_key, serper_api_key, wolfram_alpha_appid],
                            outputs=[chain_state, express_chain_state, memory_state])

    with gr.Row():
        gr.HTML("""<center>
        Powered by <a href='https://github.com/ninehills/Chat-GPT-Enhance'>Chat-GPT-Enhance</a></center>""")


    message.submit(chat_wrapper, inputs=[message, history_state, chain_state, trace_chain,
                                         monologue, express_chain_state, translate_to, force_translate],
                   outputs=[chatbot, history_state, message])

    submit.click(chat_wrapper, inputs=[message, history_state, chain_state, trace_chain,
                                         monologue, express_chain_state, translate_to, force_translate],
                 outputs=[chatbot, history_state, message])

block.launch(debug=True, share=False)
