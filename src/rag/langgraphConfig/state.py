from typing import Literal, TypedDict
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    approval_status: Literal["pending", "approved", "rejected"] | None

class AgentContext(TypedDict):
    customer_number: str
    user_id: str
    department: str
    approval_status: Literal["pending", "approved", "rejected"] | None