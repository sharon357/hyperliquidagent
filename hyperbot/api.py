"""HTTP API. Run with `hyperbot api`.

GET  /v1                  list operations
GET  /v1/{operation}      read-only operations, parameters in the query string
POST /v1/{operation}      any operation, parameters as a JSON body

If HL_API_TOKEN is set, every request needs `Authorization: Bearer <token>`.
Without a token the server only accepts being bound to localhost.
"""

from __future__ import annotations

import hmac

from fastapi import Body, Depends, FastAPI, HTTPException, Request

from .service import OPERATIONS, Service


def create_app(service: Service | None = None, token: str = "") -> FastAPI:
    svc = service or Service()
    app = FastAPI(title="hyperbot", version="0.1.0", description=__doc__)

    def auth(request: Request) -> None:
        if not token:
            return
        given = request.headers.get("authorization", "")
        if not hmac.compare_digest(given, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="missing or wrong bearer token")

    def run(name: str, args: dict) -> dict:
        if name not in OPERATIONS:
            raise HTTPException(status_code=404, detail=f"unknown operation {name!r}")
        return svc.call(name, args)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "mode": svc.s.mode}

    @app.get("/v1", dependencies=[Depends(auth)])
    def index() -> dict:
        return svc.describe()

    @app.get("/v1/{name}", dependencies=[Depends(auth)])
    def read(name: str, request: Request) -> dict:
        if name in OPERATIONS and OPERATIONS[name][0]:
            raise HTTPException(status_code=405, detail=f"{name} changes the account: use POST")
        return run(name, dict(request.query_params))

    @app.post("/v1/{name}", dependencies=[Depends(auth)])
    def write(name: str, body: dict | None = Body(default=None)) -> dict:
        return run(name, body or {})

    return app
