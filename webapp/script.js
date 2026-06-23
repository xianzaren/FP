function hideImageCard(img) {
  const card = img.closest('.image-card');
  if (card) {
    card.style.display = 'none';
  }
}

async function loadMetricBlock(task) {
  const metricsContainer = document.getElementById(`${task}-metrics`);
  const resultPath = `../output/colorectal_exp_auto/${task}/results_raytuned_vit_final.json`;
  try {
    const response = await fetch(resultPath);
    if (!response.ok) throw new Error('Result JSON not available');
    const data = await response.json();
    const { test_accuracy, test_f1_macro, test_roc_metrics, test_pr_metrics } = data;
    const html = `
      <p><strong>最终测试准确率：</strong>${(test_accuracy ?? 'N/A').toFixed(4)}</p>
      <p><strong>最终测试 Macro-F1：</strong>${(test_f1_macro ?? 'N/A').toFixed(4)}</p>
      <p><strong>ROC AUC：</strong>${test_roc_metrics?.roc_auc ?? test_roc_metrics?.roc_auc_macro_ovr ?? 'N/A'}</p>
      <p><strong>PR AUC：</strong>${test_pr_metrics?.average_precision_macro_ovr ?? test_pr_metrics?.pr_auc_tumor ?? 'N/A'}</p>
    `;
    metricsContainer.innerHTML = html;
  } catch (error) {
    metricsContainer.innerHTML = '<p>未找到结果 JSON，请确认已运行模型训练并生成输出。</p>';
  }
}

loadMetricBlock('multiclass');
loadMetricBlock('binary');
