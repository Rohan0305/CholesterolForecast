import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [formData, setFormData] = useState({
    age: '',
    height: '',
    weight: '',
    systolic_bp: '',
    diastolic_bp: '',
    glucose: '',
    exercise: '',
    smoker: '',
    gender: '',
  });
  const [predictedRange, setPredictedRange] = useState([]);


  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Form Data Submitted:", formData);  
  
    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      if (response.data.confidence_interval) {
        const { confidence_interval } = response.data;
        setPredictedRange([confidence_interval[0], confidence_interval[1]]);
      } else {
        throw new Error("No confidence interval received.");
      }
    } catch (error) {
      console.error('Error occurred while predicting cholesterol:', error);
    }
  };
  

  return (
    <div className="App" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: 'orange', minHeight: '100vh'}}>
      <h1>Cholesterol Forecast</h1>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', fontSize: '20px'}}>
        <label style={{marginBottom: "8px"}}>
          Age:
          <input type="text" name="age" value={formData.age} onChange={handleChange} />
        </label>
        <label style={{marginBottom: "8px"}}>
          Height (cm):
          <input type="text" name="height" value={formData.height} onChange={handleChange} />
        </label >
        <label style={{marginBottom: "8px"}}>
          Weight (kg):
          <input type="text" name="weight" value={formData.weight} onChange={handleChange} />
        </label>
        <label style={{marginBottom: "8px"}}>
          Systolic Blood Pressure:
          <input type="text" name="systolic_bp" value={formData.systolic_bp} onChange={handleChange} />
        </label>
        <label style={{marginBottom: "8px"}}>
          Diastolic Blood Pressure:
          <input type="text" name="diastolic_bp" value={formData.diastolic_bp} onChange={handleChange} />
        </label>
        <label style={{marginBottom: "8px"}}>
          Glucose(mg/dL)
          <input type="text" name="glucose" value={formData.glucose} onChange={handleChange} />
        </label>
        <label style={{marginBottom: "8px"}}>
          Exercise(hours/week)
          <input type="text" name="exercise" value={formData.exercise} onChange={handleChange} />
        </label>
        <label style={{marginBottom: "8px"}}>
          Smoker:
          <select name="smoker" value={formData.smoker} onChange={handleChange}>
            <option value="">Select</option>
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </label>
        <label style={{marginBottom: "8px"}}>
          Gender:
          <select name="gender" value={formData.gender} onChange={handleChange}>
            <option value="">Select</option>
            <option value="1">Male</option>
            <option value="0">Female</option>
          </select>
        </label>
        <button style={{width: "200px", height: "50px", borderRadius: "6px", margin: "auto"}} onClick={handleSubmit}>Predict Cholesterol</button>
      </form>
      {predictedRange && (
        <div style={{margin:"auto", fontSize: '18px', justifyContent: 'center'}}>
          <h3>Predicted Cholesterol Range:</h3>
          <p>Lower Bound: {predictedRange[0].toFixed(1)}</p>
          <p>Upper Bound: {predictedRange[1].toFixed(1)}</p>
        </div>
      )}
    </div>
  );
}

export default App;