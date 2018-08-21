# 简介

将[QANet](https://github.com/NLPLearn/QANet)修改为了中文可用程序。

# Quick Start

1. 配置环境。
2. 下载fasttext中文词向量并解压放置在`data/fasttext`下。
3. 将输入文件加工为以下格式：
```json
{
    "article_content":"",
    "article_id":"",
    "article_title":"",
    "article_type":"",
    "questions":[
        {
            "questions_id":"",
            "question":"",
            "answer":"",
            "question_type":""
        },
    ]
}
```
4. 依次执行`z1_load_data.py`，`z2_train.py`，`z3_evaluation.py`。
