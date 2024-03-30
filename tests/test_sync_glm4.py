from zhipuai import ZhipuAI
client = ZhipuAI(api_key="2200140b76abf3930bc2824cbbaf1310.NeYsf05Dfo7R1z3Y") 
response = client.chat.completions.create(
    model="GLM-4", 
    messages=[


        {"role": "assistant", "content": "我是人工智能助手"},
        {"role": "user", "content": "你叫什么名字"},
        {"role": "assistant", "content": "我叫chatGLM"},
        {"role": "user", "content": "你都可以做些什么事"}
    ],
)
print(response.choices[0].message)