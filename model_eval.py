import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torchvision.transforms as transforms

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, annotations, transform=None):
        """
        root_dir: Directory with all the images.
        annotations: Dictionary with key as image filename and value as annotation (bbox and class).
        transform: Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = list(self.root_dir.glob('1.jpg'))  # Adjust if using another image format
        self.annotations = annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        annotation = self.annotations[str(img_path.name)]
        return image, annotation

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
annotations = {
    "image1.jpg": {'bbox': [100, 100, 200, 200], 'class': 'cat'}
    # Add your annotations here
}
dataset = CustomDataset(root_dir='path_to_your_test_images', annotations=annotations, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the YOLO model
model = torch.load('best_2.pt')
model.eval()  # Set the model to evaluation mode

# Define the IoU calculation function
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Define the evaluation function
def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ious = []
    for inputs, gt in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)  # Adjust if your model output format differs
        
        # Simulating output, replace with actual model output handling
        pred_bbox = outputs['boxes'].detach().cpu().numpy()[0]  # Placeholder
        gt_bbox = np.array(gt['bbox'])  # Placeholder
        
        iou = calculate_iou(pred_bbox, gt_bbox)
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    return mean_iou

# Run evaluation
mean_iou = evaluate(model, dataloader)
print("Mean IoU:", mean_iou)
