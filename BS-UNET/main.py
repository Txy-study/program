import torch
from torchvision.transforms import transforms as T
import argparse #argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
from torch import optim
# from dataset import LiverDataset
from dataload import LiverDataset
from torch.utils.data import DataLoader
from model import UNet
import torch.nn.functional as F
# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import loss as L
import matplotlib.pyplot as plt
from temp import *
x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    #T.Normalize([0.5 for i in range(125)], [0.5 for i in range(125)])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])#0.5 for i in range(通道数)
# mask只需要转换为tensor
y_transform = T.ToTensor()


def train_model(model, criterion, optimizer, dataload, num_epochs,path):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y in dataload:  # 分100次遍历数据集，每次遍历batch_size=2
            x = x.float()
            y = y.float()
            inputs = x.to(device)
            labels = y.to(device)
            inputs=inputs.permute(0,2,1,3)
            # print(inputs.shape)
            # print(labels.shape)
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            outputs = model(inputs) # 前向传播
            # print(outputs.shape)
            # print(labels.shape)
            labels=labels.squeeze(1)
            # print("labels:"+str(labels.size()))
            loss = criterion(outputs, labels.long())  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        with open('train-loss_%d'%path,'a') as f:
            each_loss="epoch %d loss:%0.3f\n" %(epoch,epoch_loss/step)
            f.write(each_loss)
        #print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    torch.save(model.state_dict(), 'weights_%d.pth' %num_epochs)  # 返回模型的所有内容
    return model


# 训练模型
# DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
def train(args):
    model = UNet(n_channels=125,n_classes=10).to(device)
    batch_size = args.batch_size
    num_epochs=args.num_epochs
    save_path=args.save_path
    criterion =L.CrossEntropyLoss2d()# 损失函数
    optimizer = optim.Adam(model.parameters()) # 梯度下降 # model.parameters():Returns an iterator over module parameters
    liver_dataset = LiverDataset("data",transform=x_transform,target_transform=y_transform)# 加载数据集
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    train_model(model, criterion, optimizer, dataloader,num_epochs,save_path)
    test(args)

def test(args):
    model=UNet(n_channels=125,n_classes=10).to(device)
    model.load_state_dict(torch.load(args.ckpt %args.num_epochs,map_location='cpu'))
    liver_dataset=LiverDataset('data',transform=x_transform,target_transform=y_transform,train=False)
    dataloaders=DataLoader(liver_dataset,batch_size=1)
    model.eval()
    with torch.no_grad():
        for x,y in dataloaders:
            x=x.permute(0,2,1,3)
            print(x.shape)
            x = x.float()
            label = y.float()
            x=x.to(device)
            label=label.to(device)
            outputs=model(x)
            # print(outputs.shape)
            label = label.squeeze(1)
            # print(label_accuracy_score(label.long(),outputs,10))






if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    parser.add_argument('--action',default="test",type=str, help='train or test')  # 添加参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ckpt', type=str, default="weights_%d.pth",help='the path of the mode weight file')
    parser.add_argument("--num_epochs", type=int, default=100, help="train  times")
    parser.add_argument("--save_path",type=int,default=1,help="第几次训练")
    args = parser.parse_args()
    if args.action == 'test':
        # train(args)
        test(args)