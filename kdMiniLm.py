import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import TextSimplificationDataset
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", "-tr", action="store_true", default=True)
parser.add_argument("--eval", "-e", action="store_true", default=False)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device=", device)

def train(student,teacher,train_loader, optimizer):
  student.train()
  teacher.eval()
  train_loss = 0
  epoch = 20

  for i in tqdm(range(epoch), desc="Epoch"):
    for batch in tqdm(train_loader, desc="Iteration"):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        student_outputs = student(input_ids=input_ids, labels=labels, attention_mask=attention_mask, decoder_attention_mask= labels_attention_mask, output_attentions=True)
        
        student_enc_attention = student_outputs.encoder_attentions[-1] # Encoder last layer attention 
        student_dec_attention = student_outputs.decoder_attentions[-1] # Decoder last layer attention
        
        student_enc_attention_dist = F.log_softmax(student_enc_attention, dim=-1)
        student_dec_attention_dist = F.log_softmax(student_dec_attention, dim=-1)

        # Forward pass through teacher model
        with torch.no_grad():
          teacher_outputs = teacher(input_ids=input_ids, labels=labels,attention_mask=attention_mask, decoder_attention_mask= labels_attention_mask, output_attentions=True)
          teacher_enc_attention = teacher_outputs.encoder_attentions[-1][:, :student_enc_attention.shape[1], :, :] # Encoder last layer attention
          teacher_dec_attention = teacher_outputs.decoder_attentions[-1][:, :student_enc_attention.shape[1], :, :] # Decoder last layer attention

          teacher_enc_attention_dist = F.softmax(teacher_enc_attention, dim=-1)
          teacher_dec_attention_dist = F.softmax(teacher_dec_attention, dim=-1)

        # Compute KL-Divergence between student and teacher last layer decoder atttention and encoder attention
        loss_enc, loss_dec = 0, 0
        for b in range(student_enc_attention_dist.shape[0]): # For each batch
          for h in range(student_enc_attention_dist.shape[1]): # For each head
            for s in range(student_enc_attention_dist.shape[2]):
              loss_enc += F.kl_div(student_enc_attention_dist[b, h, s, :], teacher_enc_attention_dist[b, h, s, :], reduction='batchmean')
              loss_dec += F.kl_div(student_dec_attention_dist[b, h, s, :], teacher_dec_attention_dist[b, h, s, :], reduction='batchmean')

        # Compute total loss
        loss = loss_enc + loss_dec

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print("Epoch: {} Loss: {}".format(i, train_loss))

    torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, f"CTRL-SIMP/saved_models/t5-small_student_{epoch}.pt")

if __name__ == "__main__":  
  student = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
  tokenizer = T5Tokenizer.from_pretrained("t5-small")

  teacher = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
  teacher.load_state_dict(torch.load("CTRL-SIMP/saved_models/t5-large28.pt", map_location=device)["model_state_dict"])

  n_gpu = torch.cuda.device_count()
  print("n_gpu=", n_gpu)
  if n_gpu > 1:
    student = nn.DataParallel(student)
    teacher = nn.DataParallel(teacher)

  student = student.to(device)
  teacher = teacher.to(device)

  optimizer = torch.optim.Adam(student.parameters(), lr=1e-5)
  
  if args.train:
    df = pd.read_csv("CTRL-SIMP/datasets/Med-EASi/processed_test_data.csv")
    y = df["Simple"].values
    X = df["Expert"].values
    train_dataset = TextSimplificationDataset(X, y, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train(student, teacher, train_loader, optimizer)
  elif args.eval:
    # Add evaluation here and show the evaluation score for teacher and student
    pass