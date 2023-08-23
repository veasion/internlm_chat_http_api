书生：https://github.com/InternLM/InternLM



HTTP API 接口 支持流式和非流式请求。



运行命令：

前台进程运行： python internlm_api.py

后台进程运行： python internlm_api.py > api.log 2>&1 &



接口文档：

```
POST http://127.0.0.1:8002/internlm/chat

Body: 
{
	"message": [
		{
			"role": "user",
			"content": "你好"
		},
		{
			"role": "assistant",
			"content": "您好，请问有什么可以帮您？"
		},
		{
			"role": "user",
			"content": "你是谁？"
		}
	],
	"stream": false,
	"top_p": 0.8,
	"temperature": 0.7,
	"max_tokens": 2048
}


Response for not stream:
{
	"code": 0,
	"data": {
		"content": "我是AI机器人"
	},
	"message": "success"
}

Response for stream:
data: {"content":"我"}

data: {"content":"是"}

data: {"content":"AI"}

data: {"content":"机器人"}
```

