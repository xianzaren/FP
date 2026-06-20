document.getElementById("image_input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  const preview = document.getElementById("preview");
  preview.innerHTML = "";
  if (!file) return;
  const img = document.createElement("img");
  img.style.maxWidth = "320px";
  img.style.maxHeight = "320px";
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
});

document.getElementById("predict_btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("image_input");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please choose an image first.");
    return;
  }

  const lang = document.getElementById("lang").value;
  const task = document.getElementById("task").value;
  const model_type = document.getElementById("model_type").value;

  const form = new FormData();
  form.append("image", file);
  form.append("lang", lang);
  form.append("task", task);
  form.append("model_type", model_type);

  const resultDiv = document.getElementById("result");
  resultDiv.style.display = "block";
  resultDiv.innerText = lang === "cn" ? "正在预测..." : "Predicting...";

  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const data = await resp.json();
    if (resp.ok) {
      resultDiv.innerText = data.text;
    } else {
      resultDiv.innerText = (data.error || JSON.stringify(data));
    }
  } catch (err) {
    resultDiv.innerText = err.toString();
  }
});
