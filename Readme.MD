### Installation

```
pip install -r requirements.txt
```

### Run server

```
python serve.py
```

### Test API

```
curl -X POST http://127.0.0.1:5000/api/search -H "Content-Type: application/json" -d '{"query": "Việt Nam vững mạnh"}'

```

### Demo
#### Search `tỷ lệ vàng`

![Test](demo/demo 1.png)

#### Search `viện bảo tàng`

![Test](demo/demo 2.png)

#### Search `tỉ phú`

![Test](demo/demo 3.png)