from flask import Flask,render_template,request
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

app = Flask(__name__)

template = """
# あなたのプロフィール
・あなたは何も知らない無知な生徒です。
・あなたの情報源はユーザーからの情報のみです。

以下の条件に従って先生に質問してください。
#条件
・もしユーザーから教えてもらったことがある場合、その内容を要約して理解したかのように装ってください。
・そのトピックにおいて重要そうなことを「先生質問です！」から始め積極的に質問してください。
・ユーザーからの情報ではわからないことを質問してください。
・質問を一つに絞ってください。

{history}
Human: {human_input}
Assistant:
"""

prompt = PromptTemplate(
    input_variables = ["history","human_input"],
    template=template
)

chatgpt_chain = LLMChain(
    llm = OpenAI(temperature = 0),
    prompt = prompt,
    verbose = True,
    memory = ConversationBufferWindowMemory(k=2)
)

@app.route('/', methods = ["GET","POST"])
def home():
    if request.method == "POST":
        question = request.form["question"]
        output = chatgpt_chain.predict(human_input = question)
        return render_template("index.html", response = output)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)