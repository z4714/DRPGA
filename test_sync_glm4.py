from zhipuai import ZhipuAI
client = ZhipuAI(api_key="2200140b76abf3930bc2824cbbaf1310.NeYsf05Dfo7R1z3Y") 
response = client.chat.completions.create(
    model="GLM-4", 
    messages=[


        {"role": "assistant", "content": "我希望你逐条帮我判断我接下来提供的内容是否和python有关。有关请回答1,无关请回答0"},
        {"role": "user", "content": "好的,请给出我需要判断的内容"},
        {"role": "assistant", "content": "Create a Java class which sorts the given array of numbers. [9, 2, 4, 3, 6, 1]"},
        {"role": "user", "content": "0"}
    ],
)
print(response.choices[0].message)