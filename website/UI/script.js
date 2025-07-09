document.addEventListener("DOMContentLoaded", () => {
    const inputFields = document.getElementById("inputFields");
  
    const featureSchema = [
      { name: "Age", label: "Age (years)", type: "range", min: 10, max: 100, unit: "years" },
      { name: "Height", label: "Height (cm)", type: "range", min: 120, max: 210, unit: "cm" },
      { name: "Weight", label: "Weight (kg)", type: "range", min: 30, max: 160, unit: "kg" },
      { name: "Gender", label: "Gender", type: "select", options: ["Male", "Female"] },
      { name: "family_history_with_overweight", label: "Family History of Overweight", type: "toggle" },
      { name: "FAVC", label: "Frequent High Calorie Food Consumption", type: "toggle" },
      { name: "FCVC", label: "Vegetable Consumption (1=Low to 3=High)", type: "range", min: 1, max: 3 },
      { name: "NCP", label: "Number of Meals per Day", type: "range", min: 1, max: 4 },
      { name: "CAEC", label: "Eating Between Meals", type: "select", options: ["no", "Sometimes", "Frequently"] },
      { name: "SMOKE", label: "Smoking Habit", type: "toggle" },
      { name: "CH2O", label: "Water Intake (liters/day)", type: "range", min: 1, max: 10, unit: "L/day" },
      { name: "SCC", label: "Calorie Monitoring", type: "toggle" },
      { name: "FAF", label: "Physical Activity (hrs/week)", type: "range", min: 0, max: 14, unit: "hrs/week" },
      { name: "TUE", label: "Screen Time (hrs/day)", type: "range", min: 0, max: 12, unit: "hrs/day" },
      { name: "CALC", label: "Alcohol Consumption", type: "select", options: ["no", "Sometimes", "Frequently"] },
      { name: "MTRANS", label: "Mode of Transportation", type: "select", options: ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"] }
    ];
  
    featureSchema.forEach(feature => {
      const wrapper = document.createElement("div");
      wrapper.classList.add("input-wrapper");
  
      const label = document.createElement("label");
      label.setAttribute("for", feature.name);
      label.textContent = feature.label;
      wrapper.appendChild(label);
  
      let input;
  
      if (feature.type === "range") {
        input = document.createElement("input");
        input.type = "range";
        input.name = feature.name;
        input.id = feature.name;
        input.min = feature.min;
        input.max = feature.max;
        input.value = Math.floor((feature.min + feature.max) / 2);
        input.step = 1;
        input.required = true;
  
        const valueDisplay = document.createElement("span");
        valueDisplay.textContent = `${input.value} ${feature.unit || ""}`;
        valueDisplay.style.marginLeft = "10px";
        valueDisplay.style.fontWeight = "bold";
  
        input.addEventListener("input", () => {
          valueDisplay.textContent = `${input.value} ${feature.unit || ""}`;
        });
  
        wrapper.appendChild(input);
        wrapper.appendChild(valueDisplay);
  
      } else if (feature.type === "select") {
        input = document.createElement("select");
        input.name = feature.name;
        input.id = feature.name;
        input.required = true;
  
        feature.options.forEach(opt => {
          const option = document.createElement("option");
          option.value = opt;
          option.textContent = opt;
          input.appendChild(option);
        });
  
        wrapper.appendChild(input);
      }
  
      else if (feature.type === "toggle") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.name = feature.name;
        input.id = feature.name;
        input.classList.add("toggle-switch");
  
        const toggleLabel = document.createElement("label");
        toggleLabel.classList.add("switch-label");
        toggleLabel.appendChild(input);
  
        const slider = document.createElement("span");
        slider.classList.add("slider");
        toggleLabel.appendChild(slider);
  
        const stateLabel = document.createElement("span");
        stateLabel.textContent = "no";
        stateLabel.classList.add("toggle-state");
  
        input.addEventListener("change", () => {
          stateLabel.textContent = input.checked ? "yes" : "no";
        });
  
        wrapper.appendChild(toggleLabel);
        wrapper.appendChild(stateLabel);
      }
  
      inputFields.appendChild(wrapper);
    });
  
    const resultDiv = document.getElementById("result");
    const form = document.getElementById("predictForm");
  
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
      
        const formData = new FormData(form);
        // Handle toggles manually
        document.querySelectorAll(".toggle-switch").forEach(tog => {
          formData.set(tog.name, tog.checked ? "yes" : "no");
        });
      
        const data = Object.fromEntries(formData.entries());
        resultDiv.textContent = "🔄 Predicting...";
        resultDiv.style.color = "#88c0d0";
      
        try {
          const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
          });
      
          const result = await response.json();
      
          let categoryMap = {
            "Insufficient_Weight": "Underweight",
            "Normal_Weight": "Healthy",
            "Overweight_Level_I": "Borderline Obese",
            "Overweight_Level_II": "Borderline Obese",
            "Obesity_Type_I": "Obese (Class 1)",
            "Obesity_Type_II": "Obese (Class 2)",
            "Obesity_Type_III": "Obese (Morbid)"
          };
          
          let displayLabel = result.prediction ? categoryMap[result.prediction] || result.prediction : null;
      
          resultDiv.innerHTML = result.prediction
            ? `Predicted Class: <strong>${result.prediction}</strong><br>Category: <strong>${displayLabel}</strong> <button id="copyBtn">📋 Copy</button>`
            : `❌ Error: ${result.error || "Something went wrong"}`;
      
          resultDiv.style.color = result.prediction ? "#a3be8c" : "#bf616a";
      
          if (result.prediction) {
            document.getElementById("copyBtn").onclick = () => {
              navigator.clipboard.writeText(`${result.prediction} (${displayLabel})`);
              document.getElementById("copyBtn").textContent = "✅ Copied";
            };
          }
      
        } catch (err) {
          resultDiv.textContent = "❌ Server error. Is Flask running?";
          resultDiv.style.color = "#bf616a";
        }
      });
      
  });