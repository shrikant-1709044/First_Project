import streamlit as st 
import torch 
from PIL import Image 
import torch.nn as nn
import torchvision.transforms as transforms

st.title("Blood Cell Classifier")
st.write("upload the cell image ")


uploaded_file = st.file_uploader("Choose a cell image...",type = ["jpg","png","jpeg"])

class BloodCellClassifier(nn.Module):
    def __init__(self):
        super(BloodCellClassifier, self).__init__()

        self.con1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.rel1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2 , stride=2)


        self.con2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.rel2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.fc1 = nn.Linear(32*56*56, 128)
        self.rel3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    def forward(self , x):
        x = self.pool1(self.rel1(self.con1(x)))
        x = self.pool2(self.rel2(self.con2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.rel3(x)
        x = self.fc2(x)

        return x

model = BloodCellClassifier()
model.load_state_dict(torch.load("blood_cell_model.pth"))
model.eval()

classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']





if uploaded_file is not None :
    image = Image.open(uploaded_file)
    st.image(image , caption= 'Uploaded Blood Cell Image' )
    st.write("Processing Image...")


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor_img = transform(image).unsqueeze(0)


    with torch.no_grad():
        outputs = model(tensor_img)
        _, prediction = outputs.max(1)
        result = classes[prediction.item()]


    st.write(f"Prediction: {result}")       







    