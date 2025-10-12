from fastapi import FastAPI
from strawberry.asgi import GraphQL
from schema import schema

app = FastAPI()
graphql_app = GraphQL(schema)

app.add_route("/graphql", graphql_app)
app.add_websocket_route("/graphql", graphql_app)
