import numpy as np
from matplotlib import pyplot as plt

def evaluate_model(model, X_test, Y_test):
    loss, acc = model.evaluate(X_test, Y_test)
    print(f"[INFO] Test Accuracy: {acc:.4f}")
    
    # Predict and visualize
    idx = np.random.randint(len(X_test))
    img = np.expand_dims(X_test[idx], axis=0)
    
    pred_mask = model.predict(img)[0]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[idx].squeeze(), cmap="gray")
    plt.title("MRI Scan")
    
    plt.subplot(1, 3, 2)
    plt.imshow(Y_test[idx].squeeze(), cmap="gray")
    plt.title("Ground Truth Mask")
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), cmap="gray")
    plt.title("Predicted Mask")
    
    plt.show()
