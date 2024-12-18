# Transformers for IMAGE generation?

If diffusion models are so expensive, why not try to generate images with transformers?

I mean, if [iGPT](https://openai.com/index/image-gpt/) was pretty good, why not? 🤷‍♂

So, my initial idea was to 'reverse' the architecture of ViT-like models.

First off, let's do some image-preprocessing steps. 

```python
class PixelQuantizer:
    def __init__(self, num_bins=16, device=None):
        self.num_bins = num_bins
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bins = torch.linspace(0, 1, num_bins).to(self.device)
    
    def quantize(self, x):
        x = x.to(self.device)
        x_expanded = x.unsqueeze(-1)
        distances = torch.abs(x_expanded - self.bins)
        return torch.argmin(distances, dim=-1)
```

Yeah. That's an pixel quantizer. It's just helpful for making the image data easier to work with.

What about some patches? (arggh... i'm a pirate now)

```python
class Patchifier:
    def __init__(self, patch_size):
        self.patch_size = patch_size
    
    def patchify(self, images):
        B, C, H, W = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size)\
                       .unfold(3, self.patch_size, self.patch_size)\
                       .reshape(B, -1, self.patch_size * self.patch_size)
        return patches
```

Simple enough!

Finally, let's define the model.

```python
class ImageTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_bins, dim, num_heads, num_layers):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size_squared = patch_size * patch_size
        self.dim = dim

        self.pixel_embeddings = nn.Embedding(num_bins, dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches * self.patch_size_squared, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_logits = nn.Linear(dim, num_bins)
        
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.pixel_embeddings(x)
        x = x + self.position_embeddings
        
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        return self.to_logits(x)
    
class ImageTransformerText(ImageTransformer):
    def __init__(self, image_size, patch_size, num_bins, dim, num_heads, num_layers, vocab_size, max_length=32):
        super().__init__(image_size, patch_size, num_bins, dim, num_heads, num_layers)
        self.text_embeddings = nn.Embedding(vocab_size, dim)
        self.max_length = max_length
        
    def forward(self, x, text_input):
        B = x.shape[0]

        text_embeddings = self.text_embeddings(text_input)
        x = x.view(B, -1)
        x = self.pixel_embeddings(x)
        x = x + self.position_embeddings

        x = torch.cat([text_embeddings, x], dim=1)
        
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = x[:, self.max_length:]
        
        return self.to_logits(x)
```

Simple - but a test for now. No-one experiments with a massive model first.


## Dataset

I'm using MNIST with a bunch of synthetic text data. 

The method for the synthetic text data is literally inserting the class (0-9) into a sentence. With a lot of templates.

## Results

The model learnt much too quickly, and the results are just... not good.

I'm not sure if it's the model, the data, or the training loop. But I'm not going to spend too much time on this.

I'll do try this later, but that's all. With enough time, I'm sure this could work. But I'm not sure if it's worth it.