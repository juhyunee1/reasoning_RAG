# ? 部署指南

## 本地部署

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd rag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

编辑 `config.py`：
```python
OPENAI_API_KEY = "your_qwen_api_key"
```

### 4. 构建数据库

```bash
python build_hybrid_reasoning_db.py
```

### 5. 启动服务

```bash
./start_web_demo.sh
```

## 服务器部署

### 1. 环境准备

```bash
# 安装 Python 3.8+
sudo apt update
sudo apt install python3 python3-pip

# 安装项目依赖
pip3 install -r requirements.txt
```

### 2. 后台运行

```bash
# 使用 nohup 后台运行
nohup python3 web_demo.py --server-name 0.0.0.0 --server-port 7201 > web_demo.log 2>&1 &

# 查看日志
tail -f web_demo.log
```

### 3. 使用 systemd 服务（推荐）

创建服务文件 `/etc/systemd/system/rag-web.service`：

```ini
[Unit]
Description=RAG Web Demo
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/rag
ExecStart=/usr/bin/python3 web_demo.py --server-name 0.0.0.0 --server-port 7201
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl enable rag-web
sudo systemctl start rag-web
sudo systemctl status rag-web
```

## Docker 部署

### 1. 创建 Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7201

CMD ["python", "web_demo.py", "--server-name", "0.0.0.0", "--server-port", "7201"]
```

### 2. 构建和运行

```bash
# 构建镜像
docker build -t rag-web .

# 运行容器
docker run -d -p 7201:7201 --name rag-web rag-web
```

## 云服务部署

### 1. 使用 Hugging Face Spaces

1. 创建 Hugging Face 账号
2. 创建新的 Space
3. 上传代码
4. 配置环境变量（API Key）

### 2. 使用 Streamlit Cloud

1. 创建 `streamlit_app.py`：
```python
import streamlit as st
from web_demo import _launch_demo, _get_args
import config

# 简化的 Streamlit 版本
st.title("? 灵枢 - 神经科学推理链生成系统")
# ... 实现界面
```

2. 部署到 Streamlit Cloud

## 性能优化

### 1. 数据库优化

```bash
# 使用 SSD 存储
# 增加内存缓存
# 定期清理日志
```

### 2. API 优化

```python
# 在 config.py 中设置
API_TIMEOUT = 60
MAX_RETRIES = 3
BATCH_SIZE = 10
```

### 3. 监控

```bash
# 监控系统资源
htop
# 监控 API 调用
tail -f web_demo.log | grep "API"
```

## 故障排查

### 1. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 端口被占用 | 7201 端口被占用 | 使用 `--server-port 8080` |
| 内存不足 | 数据库太大 | 增加内存或使用 SSD |
| API 失败 | 网络或 Key 问题 | 检查网络和 API Key |
| 权限错误 | 文件权限问题 | `chmod +x start_web_demo.sh` |

### 2. 日志分析

```bash
# 查看错误日志
grep "ERROR" web_demo.log

# 查看 API 调用
grep "API" web_demo.log

# 查看性能
grep "Generation completed" web_demo.log
```

## 安全建议

1. **API Key 安全**：
   - 使用环境变量存储 API Key
   - 不要将 Key 提交到 Git

2. **网络安全**：
   - 使用防火墙限制访问
   - 考虑使用 HTTPS

3. **数据安全**：
   - 定期备份数据库
   - 设置访问权限

## 扩展功能

### 1. 多用户支持

```python
# 添加用户认证
# 实现会话管理
# 添加权限控制
```

### 2. 批量处理

```python
# 支持批量问题处理
# 添加队列系统
# 实现进度跟踪
```

### 3. 数据分析

```python
# 添加使用统计
# 实现性能监控
# 生成使用报告
```
