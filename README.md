A minimal, practical guide to using a project with a `uv.lock` file.

---

## Using a Python project with `uv.lock`

### 1. Install `uv` (if not already)

```bash
pip install uv
```

or (recommended)

```bash
brew install uv
```

---

### 2. Go to the project folder

```bash
cd your-project
```

Make sure you see:

```
pyproject.toml
uv.lock
```

---

### 3. Create environment + install dependencies

```bash
uv sync
```

This will:

* Create a virtual environment (`.venv`)
* Install exact versions from `uv.lock`

---

### 4. Activate the environment

**Mac/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```


Install requirement
```bash
pip install -r requirements.txt
```



run the code

```bash
streamlit run app.py
```



