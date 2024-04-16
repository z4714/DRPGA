import models.gpms.tiny_gpt as tiny_gpt
import llm_tools
import torch


batch_size = 16
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters=200
n_embd =384
n_head = 6    
n_layer = 6
dropout = 0.2
with open('','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)




model = tiny_gpt.GPTLM(vocab_size, )
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)




loss_best = 10
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = llm_tools.estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb,yb = llm_tools.get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter%500==0:
        eval_loss = llm_tools.estimate_loss()
        if eval_loss['val']<loss_best:
            loss_best=eval_loss['val']
            model_path = 'model_{}_{loss:.2f}.pth'.format(iter, loss_best.item())
        torch.save(model.state_dict(), model_path)
