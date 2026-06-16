"""Testa os endpoints da API. Requer uvicorn rodando em localhost:8000."""
import sys
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0


def get(path):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        return None, str(e)


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        return None, str(e)


def ok(msg):
    global PASS
    print(f"  [PASS] {msg}")
    PASS += 1


def fail(msg):
    global FAIL
    print(f"  [FAIL] {msg}")
    FAIL += 1


# GET /
status, body = get("/")
if status == 200 and "api" in body:
    ok("GET / retornou 200 com campo 'api'")
else:
    fail(f"GET / -- status={status}")

# GET /health
status, body = get("/health")
if status == 200 and body.get("modelo_carregado") is True:
    ok(f"GET /health -- modelo_carregado=True, tipo={body.get('tipo_classificador')}")
else:
    fail(f"GET /health -- status={status} body={body}")

# GET /mlflow/experimentos
status, body = get("/mlflow/experimentos")
if status == 200 and "runs" in body:
    ok(f"GET /mlflow/experimentos -- {body.get('total_runs')} runs encontrados")
else:
    fail(f"GET /mlflow/experimentos -- status={status}")

# POST /predict (payload valido)
payload = {
    "grid": 3, "round": 5, "year": 2023,
    "track_temp": 35.0, "air_temp": 28.0, "humidity": 45.0,
    "rainfall": 0.0, "wind_speed": 12.0,
    "constructor_id": 9, "circuit_id": 14,
}
status, body = post("/predict", payload)
if status == 200 and "top10" in body and "probabilidade" in body and "explicacao" in body:
    ok(f"POST /predict -- top10={body['top10']}  probabilidade={body['probabilidade']}")
    print(f"    explicacao: {body['explicacao'][:80]}...")
else:
    fail(f"POST /predict -- status={status} body={body}")

# POST /predict (payload invalido -- deve dar 422)
status, _ = post("/predict", {"grid": "invalido"})
if status == 422:
    ok("POST /predict rejeita payload invalido (HTTP 422)")
else:
    fail(f"POST /predict payload invalido -- esperado 422, recebido {status}")

print()
total = PASS + FAIL
if FAIL == 0:
    print(f"  API OK -- {PASS}/{total} testes passaram.")
else:
    print(f"  {FAIL} FALHA(S) -- {PASS}/{total} testes passaram.")
sys.exit(0 if FAIL == 0 else 1)
