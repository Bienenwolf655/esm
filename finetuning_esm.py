import argparse
import pathlib
from test_esm.pretrained import esm1b_t33_650M_UR50S
import torch.optim as optim
import time
import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import esm

from test_esm import G, FastaBatchedDataset, esm_1b_emb, pretrained


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )
    parser.add_argument(
        "epochs",
        type=int,
        help="number of epochs to train the model for",
    )
    parser.add_argument(
    	"--checkpoint",
    	type = str,
    	help = "starting training from a previous checkpoint",
    ) 
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences longer than 1024 to match the training setup",
    )
    return parser

def train(model, dataloader, optimizer,criterion = nn.CrossEntropyLoss(), log_interval = 200):

    if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
 
    model.train()  
    running_loss = 0.0
    batch = 0

    for inputs, _, _ in tqdm(dataloader, ncols=120):
    
        if args.truncate:
            inputs = inputs[:, :1022]
        since = time.time()
        batch += 1
        input = inputs.clone()
        rand_value = torch.rand(input.shape)
        rand_mask = (rand_value < 0.15) *  (input != 0) *(input != 8969)
        mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
        input = input.flatten()
        input[mask_idx] = 1
        input = input.view(inputs.size())
        if torch.cuda.is_available():
            target = inputs.to('cuda')
            input = input.to('cuda')

        for param in model.parameters():
            param.grad = None

        outputs = model(input)
        outputs = outputs['logits']
        loss = criterion(outputs.view(-1, 8970), target.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * inputs.size(0)
        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - since) * 1000 / log_interval
            loss_log = (running_loss/(batch*inputs.size(dim=0)))
            num_batches = len(dataloader)
            print(f'|{batch:5d}/{num_batches:5d} batches | '
                  f'ms/batch {ms_per_batch:5.2f} | loss {loss_log:5.2f}')
            del loss_log

        del loss, outputs, _

    epoch_loss = running_loss / len(dataloader.dataset)
    print('Training Loss: {:.4f}'.format(epoch_loss))
    del epoch_loss, running_loss
            
def evaluate(model, eval_dataloader, criterion = nn.CrossEntropyLoss() ):

    if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
    model.eval() 
    total_loss = 0.0
    with torch.no_grad():
        for inputs,_,_ in tqdm(eval_dataloader, ncols=120):
            if args.truncate:
                inputs = inputs[:, :1022]
            input = inputs.clone()
            rand_value = torch.rand(input.shape)
            rand_mask = (rand_value < 0.15) * (input != 0) *(input != 8969)
            mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input[mask_idx] = 1
            input = input.view(inputs.size())
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                input = input.to('cuda')
            output = model(input)
            output_flat = output['logits'].view(-1, 8970)
            total_loss += eval_dataloader.batch_size * criterion(output_flat, inputs.view(-1)).item()
            del output, input
            
    return total_loss / (len(eval_dataloader.dataset) - 1)

def run(model, dataloaders, epochs, optimizer, model_path=None):
    

    best_val_loss = float('inf')
    #best_model = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        epoch_start_time = time.time()
        train(model, dataloaders['train'],optimizer)
        print('-' * 10)
        val_loss = evaluate(model, dataloaders['val'])
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
           # best_model = copy.deepcopy(model.state_dict())
            if model_path is not None:
                torch.save(model.state_dict(), os.path.join(model_path,f"model_{epoch}.pt"))


def main(args):

    if os.path.exists(args.output_dir) == False:
        os.mkdir(args.output_dir)
        print('Created output directory')
    net, alphabet = esm.pretrained.load_model_and_alphabet_local(args.model_location)
    parser_model = argparse.ArgumentParser()
    esm_1b_emb.EmbBenchProtBert.add_args(parser_model)
    args_model = parser_model.parse_args([])
    model = esm_1b_emb.EmbBenchProtBert(args_model,alphabet)
    #params_optm = [{'params':model.layers[-1].parameters()},{'params':model.lm_head.scale_back.parameters()}]
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
        print('Restored model from checkpoint')
    else:
        print(f"Retrieving the weights from the pretrained ESM Model:{args.model_location}")
        model.layers.load_state_dict(net.layers.state_dict(), strict=False)
        model.lm_head.dense.load_state_dict(net.lm_head.dense.state_dict(), strict=True)
        model.lm_head.layer_norm.load_state_dict(net.lm_head.layer_norm.state_dict(), strict=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layers[-1].parameters():
        param.requires_grad = True  
    #for param_opt in params_optm:
    #    for param in param_opt['params']:
    #          param.requires_grad = True
    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    lenght = [int(0.8*len(dataset)),int(len(dataset)-int(0.8*len(dataset)))]
    train_set, val_set = torch.utils.data.random_split(dataset, lenght)
    train_load = torch.utils.data.DataLoader(train_set, collate_fn=G(), batch_size = 27)
    val_load = torch.utils.data.DataLoader(val_set, collate_fn=G(), batch_size = 27)
    dataloaders = {'train':train_load, 'val':val_load}
    optimizer=torch.optim.Adam(model.layers[-1].parameters(), lr=10**(-5), betas=(0.9, 0.999))
    #optimizer=torch.optim.Adam(params_optm, lr=10**(-5), betas=(0.9, 0.999))
    run(model, dataloaders, args.epochs, optimizer,args.output_dir)


if __name__  == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
