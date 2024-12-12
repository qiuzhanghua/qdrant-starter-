# Qdrant Starter

### 0. Run Qdrant
Open another terminal
```bash
#  tdp i plugin/qdrant_1.12.5_linux_amd64_gnu.zip 
cd $TDP_HOME/data/qdrant/1.12.5
qdrant
```

### 1. convert json to vector
```bash
wget https://storage.googleapis.com/generall-shared-data/startups_demo.json

pip install sentence-transformers numpy pandas tqdm

python convert.py
```
### 2. embed into Qdrant
```bash
python embed.py
```
### 3. 测试
```bash
python main.py
```

### 4. 运行
```zsh
pip install fastapi uvicorn
pip install httpie

python service.py
```

Open another terminal with zsh
```zsh
http GET http://localhost:8000/api/search\?q=Scout
```

### Reference
https://qdrant.tech/documentation/beginner-tutorials/neural-search/