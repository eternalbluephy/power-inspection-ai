import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000'; // Make sure this matches your backend

export const api = {
    getModels: async () => {
        const response = await axios.get(`${API_BASE_URL}/models`);
        return response.data;
    },
    detectImage: async (file: File, modelName: string) => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await axios.post(`${API_BASE_URL}/detect/image`, formData, {
            params: { model_name: modelName },
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },
    getStreamUrl: (modelName: string) => {
        // For WebSocket, we don't fetch, just return URL
        return `ws://localhost:5000/detect/stream?model_name=${modelName}`;
    }
};
