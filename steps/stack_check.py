from zenml.client import Client

# Check active stack and experiment tracker
client = Client()
active_stack = client.active_stack
print(active_stack())
