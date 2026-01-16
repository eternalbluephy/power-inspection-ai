import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import ImageDetection from "./pages/ImageDetection";
import VideoDetection from "./pages/VideoDetection";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/image" replace />} />
          <Route path="image" element={<ImageDetection />} />
          <Route path="video" element={<VideoDetection />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
