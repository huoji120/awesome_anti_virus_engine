## Preface

**key08 Security** has surpassed **3,000 followers**, meaning that a significant portion of cybersecurity professionals in China are keeping an eye on it. So, it's time for a big project.

### Why This Project?
While working in the domestic cybersecurity field, I realized that **there is still a lot of untapped potential in the overall technical level**. Many people working in cybersecurity might also be interested in how **security software** on their computers actually works. Additionally, some might even dream of developing their **own antivirus software** or see it as their long-term goal.

So, I felt there was a need to systematically **document the working principles of an antivirus engine**. While working on this, I noticed that the **information available online is close to zero**. The few available sources only describe outdated technologies like **signature-based scanning and cloud antivirus from before 2006**. Antivirus software seems to be treated like a **black box**.

To **systematically educate**, rather than spread **misinformation or meme-based security practices** like some other public security accounts, I spent **two days** developing an antivirus engine that aligns with **modern security practices (as of 2025)**.

Now, I will explain **how it works, what its weaknesses are**, and at the end of the chapter, I will even **open-source the code**, which can be **compiled directly using Visual Studio**, making **learning more convenient**.

> ⚠️ **WARNING:** This code is provided **for learning purposes only**. The **datasets for machine learning, signature analysis, and dynamic behavior detection are extremely small**, so **detection effectiveness is very limited**.
> 
> **Do not use this code for your "bypass AV" tests** and then complain that it fails to detect certain samples. This is **not intended for antivirus evasion testing**.
> **If you want to improve it, study the issues yourself instead of copying and pasting the code and then asking why it doesn't work!**

---

## Classification of Antivirus Engines
Currently, all major security vendors promote their so-called **NGAV (Next-Gen Antivirus)**, but in reality, most detection engines fall into these four categories:

1. **Cloud-Based Detection**
   - This includes:
     - **Fuzzy hashing engines** (such as `ssdeep`, `simhash`, etc.), which are used to **compare the similarity of files** (some vendors call this **"virus DNA"**).
     - **Traditional hash-based engines**, which rely on **SHA1, SHA256**, etc.
     - **Various cloud-based sandbox, manual or automated analysis systems**.

2. **Signature-Based Detection**
3. **AI & Machine Learning-Based Detection**
4. **Heuristic-Based Sandbox Detection**

Cloud-based engines are **extremely complex** and are typically a **core capability of each security company**, so **we won't discuss their implementation here** (except for those who simply use **VirusTotal (VT) as their cloud engine**). 

That leaves **categories 2, 3, and 4**, which are typically combined in AV solutions.

Each has its own strengths and weaknesses:
- **Signature-Based Detection**: Does **not** have heuristic capabilities and **fully relies on manual rule creation**, but it is the **most effective**. Each security vendor's detection capabilities **heavily rely on their signature database**.
- **Heuristic-Based Sandbox Detection**: Has **weak detection capabilities**, is **easily bypassed**, and **lags behind evolving threats**. It also tends to generate **false positives**.
- **AI/Machine Learning-Based Detection**: Provides **high detection rates** but also produces **high false positive rates**, often **negatively impacting business operations** (e.g., compiling a simple **Hello World!** application in **Visual Studio** might trigger an alert). **Many AI-based engines are overly aggressive** and flag almost anything **without a digital signature**.

---

## What Are We Going to Build?
Today, we will create **a combined Machine Learning + Behavior-Based Sandbox Engine**.

We are **not** implementing a **signature-based engine** because that would be **too simple** (if you're interested in signature matching, check out **YARA**).

The overall engine structure is as follows:
![](https://key08.com/usr/uploads/2025/03/926716651.png)

We need to implement **two core modules**:
1. **Sandbox Behavior Analysis Module**
2. **Machine Learning-Based Detection Module**

We will **introduce each module step by step**.

---

## Sandbox Module
A **sandbox module** is typically used for **unpacking and behavior analysis**. Essentially, it is a **PE file emulator**.

In our system, we use **Unicorn Engine** to **simulate CPU execution**. **Unicorn Engine** is a **lightweight**, **cross-platform** CPU emulation framework that **supports multiple architectures**, including **MIPS, ARM, PowerPC, x86, and x64**. It is based on **QEMU** and was first introduced at **Black Hat 2015** by the **GrayShift security team**.

### Main Steps of the Sandbox:
1. **Initialize the Emulation Environment**
   - Relocate PE file sections
   - Setup stack memory
   - Initialize `Unicorn Engine` and allocate virtual memory
   - Map the PE file into the virtual environment
   - Load required DLLs into the virtual machine
   - Hook critical DLL functions to monitor behavior
   - Set up essential handles, stack, **PEB**, **TEB**, etc.
   - Store important PE metadata for unpacking

2. **Relocation Processing**
   - If a **PE header contains a relocation table**, Windows will relocate **resources and functions** before execution.

3. **Memory and Stack Allocation**
   - The **stack memory** must be fully emulated for the execution environment.

4. **Mapping PE Sections into Memory**
   - A **PE file's size on disk differs from its actual size when loaded in memory**.
   - We must **expand** it and **map each section accordingly**.

5. **Load Required DLLs**
   - **Parse the Import Table** and **map necessary DLLs** into our virtual machine.

6. **Intercept API Calls**
   - Hook **imported API functions**.

7. **Shellcode & Packed Malware Detection**
   - Monitor for **self-modifying code execution**, which indicates **packed malware**.

8. **Behavior-Based Detection**
   - Detect suspicious behavior, such as:
     - **Downloading executable files via `WinHttp`**
     - **Excessive `sleep` delays**
     - **Accessing sensitive directories**
     - **Direct access to `LDR` structures** (used to detect stealth malware)

### Sandbox Performance:
Here’s an example detection result:
![](https://key08.com/usr/uploads/2025/03/408250478.png)

---

## Machine Learning Module
The **machine learning module** is used to classify files based on extracted PE features.

### Feature Engineering:
We extract the following feature sets:
1. **PE Header Features** (Presence of Import Tables, TLS sections, relocations, etc.)
2. **Imported DLLs** (Checks for specific suspicious DLLs)
3. **File Entropy** (Measures randomness)
4. **Entry Point Byte Sequence** (Examines the first 64 bytes of code)
5. **Section Analysis** (Checks PE section sizes and entropy)
6. **Code-to-Data Ratio** (Compares code section size vs. total PE file size)

### Training Data:
We collected **1,000 benign samples** and **1,000 malicious samples**, saved their features into a **CSV file**, and used them for training.

![](https://key08.com/usr/uploads/2025/03/1410311475.png)

> ⚠️ **NOTE:** The dataset is **too small** for real-world performance. A proper dataset should have at least **100,000+ benign and 100,000+ malicious samples**.

### Model Training:
We use **XGBoost** for training and then export the trained model to **pure C++ code** using **m2cgen**.

![](https://key08.com/usr/uploads/2025/03/358391058.png)

---

## Conclusion
This is a **basic but modern antivirus engine** using **sandbox-based behavior analysis** and **machine learning-based detection**.

The **full source code** is available on **GitHub** (link below). 🚀

🔗 **GitHub Repository:** [INSERT LINK HERE]