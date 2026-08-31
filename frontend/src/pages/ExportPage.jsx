import React from 'react';
import { Download, FileJson, FileSpreadsheet, Database } from 'lucide-react';

export default function ExportPage({ predictionTensor, modelDepths, date }) {
  
  const handleDownloadJSON = () => {
    if (!predictionTensor) return alert('No data to download. Please run a prediction first.');
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(predictionTensor));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `ocean_prediction_${date}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const handleDownloadCSV = () => {
    if (!predictionTensor) return alert('No data to download. Please run a prediction first.');
    
    // Flatten the tensor (15, lat, lon) into a CSV
    // This is a simplified CSV generator for the PoC
    let csvContent = "data:text/csv;charset=utf-8,Depth_Level,Lat_Index,Lon_Index,Temperature\n";
    
    const depths = predictionTensor[0]; // [15, lat, lon]
    depths.forEach((layer, d_idx) => {
      const depth_m = modelDepths[d_idx] || d_idx;
      layer.forEach((row, lat_idx) => {
        row.forEach((val, lon_idx) => {
          if (val !== null) {
            csvContent += `${depth_m},${lat_idx},${lon_idx},${val}\n`;
          }
        });
      });
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `ocean_prediction_${date}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const handleDownloadNetCDF = () => {
    alert('NetCDF export requires backend processing. For the PoC, please use JSON or CSV export.');
  };

  return (
    <div className="flex-1 p-8 overflow-y-auto">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl font-display font-bold text-[var(--color-ink-dark)] mb-2">Data Export Hub</h2>
        <p className="text-[var(--color-ink-medium)] mb-8">
          Download the latest 3D thermal reconstruction tensor for offline analysis or data assimilation.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* JSON Export */}
          <div className="bg-white border border-[var(--color-paper-border)] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-lg flex items-center justify-center mb-4">
              <FileJson className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold mb-2">JSON Format</h3>
            <p className="text-sm text-[var(--color-ink-medium)] mb-6">
              Complete raw tensor array in JSON format. Best for web applications and quick parsing.
            </p>
            <button 
              onClick={handleDownloadJSON}
              className="w-full py-2 bg-blue-50 text-blue-700 rounded-lg font-semibold hover:bg-blue-100 transition-colors flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" /> Download JSON
            </button>
          </div>

          {/* CSV Export */}
          <div className="bg-white border border-[var(--color-paper-border)] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-green-50 text-green-600 rounded-lg flex items-center justify-center mb-4">
              <FileSpreadsheet className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold mb-2">CSV Format</h3>
            <p className="text-sm text-[var(--color-ink-medium)] mb-6">
              Flattened tabular data (Depth, Lat, Lon, Temp). Ideal for Excel, Pandas, or MATLAB.
            </p>
            <button 
              onClick={handleDownloadCSV}
              className="w-full py-2 bg-green-50 text-green-700 rounded-lg font-semibold hover:bg-green-100 transition-colors flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" /> Download CSV
            </button>
          </div>

          {/* NetCDF Export */}
          <div className="bg-white border border-[var(--color-paper-border)] rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow opacity-75">
            <div className="w-12 h-12 bg-purple-50 text-purple-600 rounded-lg flex items-center justify-center mb-4">
              <Database className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold mb-2">NetCDF (.nc)</h3>
            <p className="text-sm text-[var(--color-ink-medium)] mb-6">
              Climate and Forecast (CF) compliant NetCDF file with complete geospatial metadata.
            </p>
            <button 
              onClick={handleDownloadNetCDF}
              className="w-full py-2 bg-purple-50 text-purple-700 rounded-lg font-semibold hover:bg-purple-100 transition-colors flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" /> Request Backend Export
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
