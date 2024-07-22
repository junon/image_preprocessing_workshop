# image_preprocessing_workshop
Developed with AI Foundations Team from Manufacturing Automations, Intel

## Setup Guide
### 1. Google Colab, simplest and minimal setup required
> Note: Requires a Gmail account to log in and connect to Google Colab  
> Note: If you have privacy concerns, please complete a local installation instead, refer below for instructions


### 2. Local installation, estimated completion time: 30 mins
1. Download latest python version from https://www.python.org/downloads/
![Download Python](/readme_images/downloadpython.png)
2. Run the downloaded executable 
   * uncheck "Add python.exe to PATH" to avoid clashing with your existing python version/workspaces. 
   * Select **"Customize installation"**  
![Customize Installation](/readme_images/customizeinstallation.png)
3. In Optional Features, 
   * choose only to install **"pip"** and 
   * click **"Next"** as follows  
![Optional Features](/readme_images/optionalfeatures.png)
4. In Advanced Options, 
   * uncheck all tickboxes, 
   * copy the **"customize install location"** path and saved it to a sticky note / notepad 
   * click **"Next"** as follows  
![Advanced Options](/readme_images/advancedoptions.png)  
In case if you lose your python installation path, most probably it will be defaulted to
    ```C:\Users\<user>\AppData\Local\Programs\Python\<PythonVersion>```
5. Once installation is completed, go ahead and close the executable.
![Setup Successful](/readme_images/setupsuccessful.png)
6. Download the repository as **"Download ZIP"**  
![Download Zip](/readme_images/downloadzip.png)
7. Extract the zip file to Desktop  
![Extract Zip](/readme_images/extractzip.png)
8. Open the directory
   * right click and select **"Open in Terminal"**
   * if **"Open in Terminal"** is not available, try selecting **"Show more options"**  
   ![Open In Terminal](/readme_images/openinterminal.png)
9. Paste the python path you've previously saved and check the version as follows
   ```
   C:\Users\<user>\AppData\Local\Programs\Python\<PythonVersion>\python.exe -V
   ```
   ![Python Version](/readme_images/pythonvenv.png)
10. Create a virtual environment (venv) by
    ```
    C:\Users\<user>\AppData\Local\Programs\Python\<PythonVersion>\python.exe -m venv venv
    ```
    * A new folder named venv would appear in the repository directory
11. Activate the virtual environment by
    ```
    .\venv\Scripts\activate
    ```
    * The terminal line would appear **(venv)** as follows  
   ![Python venv](/readme_images/pythonvenv.png)
12. **Ensure (venv) is activated** and pip install the necessary python packages / libraries / dependencies by
    ```
    pip install -r .\requirements.txt -i https://devpi.intel.com/root/pypi
    ```
    ![Pip Install](/readme_images/pipinstall.png)
    * Once pip installation has been completed, it should show all the libraries successfully installed as follows  
![Pip Done](/readme_images/pipinstalldone.png)
13. **Ensure (venv) is activated** and we are ready to run the code by 
    ```
    python .\ai_summit_opencv_workshop.py
    ```
    ![Run Code](/readme_images/runcode.png)
    * You should be able to see a window appear as follow  
    ![Testing Window](/readme_images/testingwindow.png)
14. **Congratulations! Setup is complete and feel free to play with the sliders.**

