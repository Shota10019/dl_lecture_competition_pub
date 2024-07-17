import re
import random
import time
from statistics import mode
import timm
from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    text = text.lower()
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, tokenizer_name='roberta-base', transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.answer = answer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.answer2idx = {}
        self.idx2answer = {}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = process_text(self.df["question"][idx])
        tokenized_input = self.tokenizer(question, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokenized_input['input_ids'].squeeze(0)
        attention_mask = tokenized_input['attention_mask'].squeeze(0)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)

            return image, input_ids, attention_mask, torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, input_ids, attention_mask

    def __len__(self):
        return len(self.df)

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

class RobertaEncoder(nn.Module):
    def __init__(self, roberta_model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state 

class VQAModel(nn.Module):
    def __init__(self, roberta_model_name='roberta-base', n_answer=1000):
        super().__init__()
        self.resnet = timm.create_model('efficientnet_b4', pretrained=True, num_classes=512)
        self.roberta_encoder = RobertaEncoder(roberta_model_name)
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(1024 + 512, 512),  
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, input_ids, attention_mask):
        image_feature = self.resnet(image)
        roberta_output = self.roberta_encoder(input_ids, attention_mask)
        lstm_output, _ = self.lstm(roberta_output)
        lstm_output = lstm_output[:, -1, :]  
        x = torch.cat([image_feature, lstm_output], dim=1)
        x = self.fc(x)
        return x

def train(model, dataloader, optimizer_roberta, optimizer_other, criterion, device, scheduler_roberta, accumulation_steps=4):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    optimizer_roberta.zero_grad()
    optimizer_other.zero_grad()
    for batch_idx, (image, input_ids, attention_mask, answers, mode_answer) in enumerate(dataloader):
        image, input_ids, attention_mask, answers, mode_answer = \
            image.to(device), input_ids.to(device), attention_mask.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, input_ids, attention_mask)
        loss = criterion(pred, mode_answer)

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer_roberta.step()
            optimizer_other.step()
            optimizer_roberta.zero_grad()
            optimizer_other.zero_grad()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer_roberta.step()
        optimizer_other.step()
        optimizer_roberta.zero_grad()
        optimizer_other.zero_grad()

    scheduler_roberta.step()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    with torch.no_grad():
        for image, input_ids, attention_mask, answers, mode_answer in dataloader:
            image, input_ids, attention_mask, answers, mode_answer = \
                image.to(device), input_ids.to(device), attention_mask.to(device), answers.to(device), mode_answer.to(device)

            pred = model(image, input_ids, attention_mask)
            loss = criterion(pred, mode_answer)

            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)
            simple_acc += (pred.argmax(1) == mode_answer).mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3/4, 4/3), interpolation=Image.BILINEAR),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    ])
    train_dataset = VQADataset(df_path="/content/data/train.json", image_dir="/content/data/train", transform=transform)
    test_dataset = VQADataset(df_path="/content/data/valid.json", image_dir="/content/data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    def collate_fn(batch):
        if len(batch[0]) == 5:  
            images, input_ids, attention_masks, answers, mode_answers = zip(*batch)
            images = torch.stack(images)
            input_ids = torch.stack(input_ids)
            attention_masks = torch.stack(attention_masks)
            answers = torch.stack(answers)
            mode_answers = torch.tensor(mode_answers)
            return images, input_ids, attention_masks, answers, mode_answers
        else:  
            images, input_ids, attention_masks = zip(*batch)
            images = torch.stack(images)
            input_ids = torch.stack(input_ids)
            attention_masks = torch.stack(attention_masks)
            return images, input_ids, attention_masks

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = VQAModel(roberta_model_name='roberta-base', n_answer=len(train_dataset.answer2idx)).to(device)

    num_epoch = 5
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model_params = list(model.named_parameters())

    roberta_params = [p for n, p in model_params if "roberta" in n]
    other_params = [p for n, p in model_params if "roberta" not in n]

    optimizer_roberta = AdamW(roberta_params, lr=2e-5, weight_decay=0.01)
    optimizer_other = AdamW(other_params, lr=2e-5 * 50, weight_decay=0.01)

    total_steps = len(train_loader) * num_epoch

    scheduler_roberta = get_linear_schedule_with_warmup(
        optimizer_roberta,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    accumulation_steps = 4 

    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer_roberta, optimizer_other, criterion, device, scheduler_roberta, accumulation_steps)
        print(f"{epoch + 1}/{num_epoch}\n"
              f"train time: {train_time:.2f}　\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    model.eval()
    submission = []
    for image, input_ids, attention_mask in test_loader:
        image, input_ids, attention_mask = image.to(device), input_ids.to(device), attention_mask.to(device)
        pred = model(image, input_ids, attention_mask)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
