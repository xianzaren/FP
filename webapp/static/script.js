const texts = {
  en: {
    title: "Colorectal Patch Classifier",
    langLabel: "Language:",
    taskLabel: "Mode:",
    modelLabel: "Model:",
    imageLabel: "Choose image:",
    predictBtn: "Predict",
    viewVisualizations: "View visualizations",
    chooseFileBtn: "Choose file",
    chooseImageFirst: "Please choose an image first.",
    noFileChosen: "No file chosen",
    predicting: "Predicting...",
  },
  cn: {
    title: "结直肠补丁分类器",
    langLabel: "语言:",
    taskLabel: "模式:",
    modelLabel: "模型:",
    imageLabel: "选择图片:",
    predictBtn: "预测",
    chooseFileBtn: "选择文件",
    viewVisualizations: "查看可视化",
    chooseImageFirst: "请先选择一张图片。",
    noFileChosen: "未选择文件",
    predicting: "正在预测...",
  },
};

const langSelect = document.getElementById("lang");
const taskSelect = document.getElementById("task");
const modelSelect = document.getElementById("model_type");
const predictBtn = document.getElementById("predict_btn");
const visualizeLink = document.getElementById("visualize_link");
const chooseFileBtn = document.getElementById("choose_file_btn");
const chosenFileName = document.getElementById("chosen_file_name");
const fileInput = document.getElementById("image_input");

function updateLanguageTexts() {
  const lang = langSelect.value;
  const current = texts[lang];
  document.getElementById("pageTitle").innerText = current.title;
  document.getElementById("langLabel").innerText = current.langLabel;
  document.getElementById("taskLabel").innerText = current.taskLabel;
  document.getElementById("modelLabel").innerText = current.modelLabel;
  document.getElementById("imageLabel").innerText = current.imageLabel;
  predictBtn.innerText = current.predictBtn;
  chooseFileBtn.innerText = current.chooseFileBtn;
  visualizeLink.innerText = current.viewVisualizations;
  if (!fileInput.files.length) {
    chosenFileName.innerText = current.noFileChosen;
  }
}

langSelect.addEventListener("change", updateLanguageTexts);
updateLanguageTexts();

chooseFileBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  const preview = document.getElementById("preview");
  preview.innerHTML = "";
  if (!file) {
    const current = texts[langSelect.value];
    chosenFileName.innerText = current.noFileChosen;
    return;
  }
  chosenFileName.innerText = file.name;
  const img = document.createElement("img");
  img.style.maxWidth = "320px";
  img.style.maxHeight = "320px";
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
});

predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  const lang = langSelect.value;
  const localeTexts = texts[lang];
  if (!file) {
    alert(localeTexts.chooseImageFirst);
    return;
  }

  const task = taskSelect.value;
  const model_type = modelSelect.value;

  const form = new FormData();
  form.append("image", file);
  form.append("lang", lang);
  form.append("task", task);
  form.append("model_type", model_type);

  const resultDiv = document.getElementById("result");
  resultDiv.style.display = "block";
  resultDiv.innerText = localeTexts.predicting;

  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const text = await resp.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
    if (resp.ok && data) {
      resultDiv.innerText = data.text;
    } else {
      resultDiv.innerText = data?.error || text || resp.statusText;
    }
  } catch (err) {
    resultDiv.innerText = err.toString();
  }
});
