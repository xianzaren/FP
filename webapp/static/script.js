const supportedArchitectures = {
  imageSizes: [128, 224, 384],
  patchSizes: [16, 32],
  vitDepths: ["tiny", "small", "base"],
};

const taskSelect = document.getElementById("task");
const modelSelect = document.getElementById("model_type");
const imageSizeSelect = document.getElementById("image_size");
const patchSizeSelect = document.getElementById("patch_size");
const vitDepthSelect = document.getElementById("vit_depth");
const vitControls = document.getElementById("vit-controls");
const experimentStatus = document.getElementById("experiment-status");
const checkpointBadge = document.getElementById("checkpoint-badge");
const imageInput = document.getElementById("image_input");
const chooseFileButton = document.getElementById("choose_file_btn");
const chosenFileName = document.getElementById("chosen_file_name");
const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict_btn");
const result = document.getElementById("result");
const startTrainingButton = document.getElementById("start_training_btn");
const trainingBadge = document.getElementById("training-badge");
const trainingMeta = document.getElementById("training-meta");
const trainingLog = document.getElementById("training-log");

let experiments = [];
let selectedExperiment = null;
let previousPreviewUrl = null;
let lastTrainingState = "idle";

function setBadge(element, label, tone = "neutral") {
  element.textContent = label;
  element.className = `status-badge ${tone}`;
}

function selectOptions(select, values, preferredValue) {
  const previousValue = String(preferredValue ?? select.value);
  select.replaceChildren();
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = String(value);
    if (String(value) === previousValue) {
      option.selected = true;
    }
    select.appendChild(option);
  });
  if (!select.value && values.length) {
    select.value = String(values[0]);
  }
}

function configureArchitectureControls() {
  selectOptions(imageSizeSelect, supportedArchitectures.imageSizes, imageSizeSelect.value || 224);
  selectOptions(patchSizeSelect, supportedArchitectures.patchSizes, patchSizeSelect.value || 16);
  selectOptions(vitDepthSelect, supportedArchitectures.vitDepths, vitDepthSelect.value || "base");
}

function currentArchitecture() {
  return {
    image_size: Number(imageSizeSelect.value),
    patch_size: Number(patchSizeSelect.value),
    vit_depth: vitDepthSelect.value,
  };
}

function resolveSelectedExperiment() {
  const architecture = currentArchitecture();
  selectedExperiment = experiments.find((experiment) => (
    Number(experiment.image_size) === architecture.image_size
    && Number(experiment.patch_size) === architecture.patch_size
    && experiment.vit_depth === architecture.vit_depth
  )) || null;
}

function updatePredictionAvailability() {
  const isVit = modelSelect.value === "vit";
  vitControls.hidden = !isVit;

  if (!isVit) {
    selectedExperiment = null;
    setBadge(checkpointBadge, "AutoKeras baseline", "neutral");
    experimentStatus.textContent = "AutoKeras uses its saved 128px model for the selected task.";
    predictButton.disabled = false;
    return;
  }

  resolveSelectedExperiment();
  if (selectedExperiment) {
    setBadge(checkpointBadge, "Checkpoint ready", "success");
    experimentStatus.textContent = `Prediction will use: ${selectedExperiment.label}.`;
    predictButton.disabled = false;
  } else {
    setBadge(checkpointBadge, "Checkpoint missing", "warning");
    experimentStatus.textContent = "No completed checkpoint matches this architecture. Start training below or choose a saved combination.";
    predictButton.disabled = true;
  }
}

async function refreshExperiments() {
  setBadge(checkpointBadge, "Loading checkpoints", "neutral");
  try {
    const response = await fetch(`/api/experiments/${encodeURIComponent(taskSelect.value)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || response.statusText);
    }
    experiments = data.experiments || [];
  } catch (error) {
    experiments = [];
    experimentStatus.textContent = `Could not load checkpoints: ${error.message}`;
  }
  updatePredictionAvailability();
}

function showResult(message, isError = false) {
  result.hidden = false;
  result.classList.toggle("error", isError);
  result.textContent = message;
}

chooseFileButton.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  preview.replaceChildren();
  if (previousPreviewUrl) {
    URL.revokeObjectURL(previousPreviewUrl);
    previousPreviewUrl = null;
  }
  if (!file) {
    chosenFileName.textContent = "Drop an image or browse";
    return;
  }

  chosenFileName.textContent = file.name;
  previousPreviewUrl = URL.createObjectURL(file);
  const image = document.createElement("img");
  image.src = previousPreviewUrl;
  image.alt = "Selected pathology patch preview";
  preview.appendChild(image);
});

predictButton.addEventListener("click", async () => {
  const file = imageInput.files[0];
  if (!file) {
    showResult("Choose an image before predicting.", true);
    return;
  }

  const isVit = modelSelect.value === "vit";
  if (isVit && !selectedExperiment) {
    showResult("Choose an architecture with a completed checkpoint first.", true);
    return;
  }

  const form = new FormData();
  form.append("image", file);
  form.append("task", taskSelect.value);
  form.append("model_type", modelSelect.value);
  if (isVit) {
    form.append("experiment_id", selectedExperiment.id);
  }

  predictButton.disabled = true;
  predictButton.textContent = "Predicting...";
  showResult("Loading model and predicting patch...");
  try {
    const response = await fetch("/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || response.statusText);
    }
    showResult(`${data.text}\nModel: ${data.model_label}`);
  } catch (error) {
    showResult(error.message, true);
  } finally {
    predictButton.textContent = "Predict selected patch";
    updatePredictionAvailability();
  }
});

function updateTrainingStatus(status) {
  const state = status.state || "idle";
  const tone = state === "running" ? "running" : state === "completed" ? "success" : state === "failed" ? "danger" : "neutral";
  const labels = {
    idle: "No active job",
    running: "Training in progress",
    completed: "Training completed",
    failed: "Training failed",
  };
  setBadge(trainingBadge, labels[state] || state, tone);
  trainingMeta.textContent = status.started_at ? `Started: ${new Date(status.started_at).toLocaleString()}` : "";
  if (typeof status.log_tail === "string") {
    trainingLog.textContent = status.log_tail || "Waiting for training output...";
    trainingLog.scrollTop = trainingLog.scrollHeight;
  }
  startTrainingButton.disabled = state === "running";

  if (lastTrainingState === "running" && state === "completed") {
    refreshExperiments();
  }
  lastTrainingState = state;
}

async function refreshTrainingStatus() {
  try {
    const response = await fetch("/api/training/status");
    const status = await response.json();
    if (response.ok) {
      updateTrainingStatus(status);
    }
  } catch (_) {
    // The rest of the UI remains usable if a status refresh is interrupted.
  }
}

startTrainingButton.addEventListener("click", async () => {
  const architecture = currentArchitecture();
  const payload = {
    task: taskSelect.value,
    ...architecture,
    num_samples: Number(document.getElementById("num_samples").value),
    tune_epochs: Number(document.getElementById("tune_epochs").value),
    final_epochs: Number(document.getElementById("final_epochs").value),
  };

  startTrainingButton.disabled = true;
  startTrainingButton.textContent = "Starting training...";
  try {
    const response = await fetch("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || response.statusText);
    }
    updateTrainingStatus(data.status);
  } catch (error) {
    showResult(`Training was not started: ${error.message}`, true);
    startTrainingButton.disabled = false;
  } finally {
    startTrainingButton.textContent = "Start training run";
  }
});

[taskSelect, modelSelect, imageSizeSelect, patchSizeSelect, vitDepthSelect].forEach((element) => {
  element.addEventListener("change", () => {
    if (element === taskSelect) {
      refreshExperiments();
    } else {
      updatePredictionAvailability();
    }
  });
});

configureArchitectureControls();
refreshExperiments();
refreshTrainingStatus();
window.setInterval(refreshTrainingStatus, 4000);
