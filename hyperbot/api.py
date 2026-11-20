"""HTTP API. Run with `hyperbot api`.

GET  /v1                  list operations
GET  /v1/{operation}      read-only operations, parameters in the query string
POST /v1/{operation}      any operation, parameters as a JSON body

If HL_API_TOKEN is set, every request needs `Authorization: Bearer <token>`.
Without a token the server only accepts being bound to localhost.

With a token, the same operations are also served as a remote MCP endpoint at
/mcp/<token>, for clients that connect by URL (such as the Claude apps). The
token in the path is the only credential, so treat that URL like a password.
"""

from __future__ import annotations

import hmac

from fastapi import Body, Depends, FastAPI, HTTPException, Request

from .service import OPERATIONS, Service


def create_app(service: Service | None = None, token: str = "") -> FastAPI:
    svc = service or Service()
    mcp_app = None
    lifespan = None
    if token:
        from contextlib import asynccontextmanager

        from mcp.server.transport_security import TransportSecuritySettings

        from .mcp_server import build_server

        mcp_server = build_server(svc)
        mcp_app = mcp_server.streamable_http_app(
            streamable_http_path=f"/mcp/{token}",
            stateless_http=True,
            json_response=True,
            # Served behind the host's proxy on a public domain, so the localhost-only Host check does not apply.
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
            host="0.0.0.0",
        )

        @asynccontextmanager
        async def lifespan(_app):
            async with mcp_server.session_manager.run():
                yield

    # A token means the server is reachable from outside: don't publish an unauthenticated map of its operations.
    docs = {"docs_url": None, "redoc_url": None, "openapi_url": None} if token else {}
    app = FastAPI(title="hyperbot", version="0.1.0", description=__doc__, lifespan=lifespan, **docs)

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

    if mcp_app is not None:
        app.mount("/", mcp_app)  # after the routes above, so it only sees paths they did not match
    return app
