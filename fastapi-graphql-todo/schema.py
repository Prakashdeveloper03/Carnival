import strawberry
from typing import List
from models import Todo

# In-memory database
db: List[Todo] = []


@strawberry.type
class TodoType:
    id: int
    text: str
    completed: bool


@strawberry.type
class Query:
    @strawberry.field
    def todos(self) -> List[TodoType]:
        return db


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_todo(self, text: str) -> TodoType:
        todo = Todo(id=len(db) + 1, text=text, completed=False)
        db.append(todo)
        return todo

    @strawberry.mutation
    def update_todo(self, id: int, completed: bool) -> TodoType:
        for todo in db:
            if todo.id == id:
                todo.completed = completed
                return todo
        raise Exception("Todo not found")

    @strawberry.mutation
    def delete_todo(self, id: int) -> bool:
        for i, todo in enumerate(db):
            if todo.id == id:
                db.pop(i)
                return True
        return False


schema = strawberry.Schema(query=Query, mutation=Mutation)
