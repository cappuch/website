# P104-100s: Affordable Inference on a Budget

For a while, I had been considering getting an NVIDIA P40 for some cost-effective large language model (LLM) inference. However, ever since the release of LLAMA-3, the price of P40s has skyrocketed by at least a hundred euros!

That led me to explore alternative options, and I discovered the P104-100s. These mining cards, produced by NVIDIA, were originally designed for CUDA workloads. And as luck would have it, AI inference also thrives on GPGPU/CUDA capabilities.

## The Quirks of P104-100s

P104-100s are peculiar cards. They physically come with 8GB of memory, but for some reason, the BIOS locks them to 4GB. Thankfully, this limitation is easily bypassed by flashing the BIOS, and most of the cards on the market have already been modified to unlock their full potential.

### The Cost Breakdown

Here’s the best part: each card costs only about 20 euros. The downside? Shipping and customs can be a pain, bringing the total cost to around 130 euros for two cards. Still, that’s a steal for the performance they offer.

## Performance Metrics

Let’s talk numbers. Here’s how the P104-100s stack up in real-world AI tasks:

- **SD-XL:** 2.52 iterations per second
- **Qwen 2 7B Coder (Q8):** 18.75 tokens per second
- **Qwen 2 13B Coder (Q6):** 12.5 tokens per second
- **Qwen 2 13B Coder (Q8):** 7 tokens per second

While these numbers aren’t groundbreaking, they’re impressive given the price point. For anyone on a budget, P104-100s offer an incredible value proposition.

## Future Plans

As much as I’m enjoying the P104-100s, I can’t help but look ahead. The P102-100s, while significantly more expensive, offer double the performance. They might just be my next upgrade when the time is right.

## Conclusion

If you’re looking for an affordable way to run large language models, the P104-100s are a fantastic option. They’re cost-effective, easy to set up, and offer decent performance. Just remember to flash the BIOS to unlock their full potential!