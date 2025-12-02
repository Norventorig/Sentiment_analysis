const btn = document.getElementById("btnPredict");
const input = document.getElementById("inputText");
const status = document.getElementById("status");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const rawOutput = document.getElementById("rawOutput");

btn.addEventListener("click", async () => {
  const text = input.value;
  status.textContent = "Отправка...";
  resultLabel.textContent = "—";
  resultConfidence.textContent = "—";
  rawOutput.textContent = "—";

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });

    const data = await resp.json();
    if (!data.success) {
      status.textContent = "Ошибка: " + (data.error || "unknown");
      rawOutput.textContent = JSON.stringify(data, null, 2);
      return;
    }

    const r = data.result;
    resultLabel.textContent = r.label ?? "—";
    resultConfidence.textContent = "Доверие: " + (r.confidence ?? "—");
    rawOutput.textContent = JSON.stringify(r, null, 2);
    status.textContent = "Готово";
  } catch (err) {
    status.textContent = "Ошибка сети";
    rawOutput.textContent = String(err);
  }
});
