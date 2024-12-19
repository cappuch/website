# W5XDE
Distributed training with PyTorch is powerful. But what about international levels worth of power? W5XDE is a cheaply written framework for accelerating PyTorch training with multiple GPUs across multiple machines. It's not as powerful as PyTorch's native distributed training, but it's a lot easier to set up and use. This is a write-up of my experience developing W5XDE and using it to train simple models.

## The Idea
The idea for W5XDE came to me when I was trying to train a large language model on a budget. I had a few GPUs lying around, and I wanted to use them to speed up training. I looked into PyTorch's native distributed training, but it seemed too complicated for my needs. I wanted something simple that I could set up quickly and start using right away. That's when I decided to write W5XDE.

## The Design
W5XDE is a simple framework that allows you to train PyTorch models on multiple GPUs across multiple machines. It consists of two main components: a master node and worker nodes. The master node is responsible for coordinating the training process, while the worker nodes are responsible for actually running the training code.

### The Master Node
The master node (server) is a simple script which distrobutes batches and recieves gradients from the workers.

```python
def distribute_batches(self):
    dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    for batch_idx, batch in enumerate(dataloader):
        processed_batch = {
            'batch_id': f"{self.global_step}_batch{batch_idx}",
            'input_ids': batch['input_ids'].cpu(),
            'labels': batch.get('labels', None)
        }
        self.batch_queue.put(processed_batch)
```

Pretty simple, right? The master node just loads the dataset, creates a dataloader, and starts distributing batches to the workers. It also listens for gradients from the workers and updates the model accordingly.

### The Worker Nodes
Worker nodes are responsible for receiving batches from the master node, processing them, and sending the gradients back to the master node.
It interestingly includes handshakes and a lot of other stuff that I'm too lazy to write about.

We designed it to be as simple as possible to set up, and it's been working great for us so far.

## Training examples
```python
from w5xde.w5xde import TrainingNode
from custom_functions import Model

if __name__ == "__main__":
    model = Model(30522) # 30522 is vocab size
    node = TrainingNode(model, secure=False)
    node.train()
```

It's that fucking simple. Just create a model, create a TrainingNode, and call the train method. W5XDE takes care of the rest.

## Conclusion
I genuinely forgot to implement gradient aggregation, so the model doesn't actually learn anything. But hey, it's a start. I'm planning to add more features in the future, so stay tuned for updates.