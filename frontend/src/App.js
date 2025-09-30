import { useEffect, useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { Sparkles, Heart, Download, Loader2, Wand2, ImageIcon } from "lucide-react";

// BACKEND URL
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8001';
const API = `${API_BASE}/api`;

const Home = () => {
  const [prompt, setPrompt] = useState("");
  const [topWallpapers, setTopWallpapers] = useState([]);
  const [generating, setGenerating] = useState(false);
  const [generatedWallpaper, setGeneratedWallpaper] = useState(null);
  const [rateLimit, setRateLimit] = useState({ used: 0, remaining: 5, limit: 5 });

  const fetchTopWallpapers = async () => {
    try {
      const response = await axios.get(`${API}/wallpapers/top`);
      if (response.data.success) {
        setTopWallpapers(response.data.wallpapers);
      }
    } catch (e) {
      console.error("Error fetching top wallpapers:", e);
    }
  };

  const checkRateLimit = async () => {
    try {
      const response = await axios.get(`${API}/wallpapers/limit/check`);
      if (response.data.success) {
        setRateLimit(response.data);
      }
    } catch (e) {
      console.error("Error checking rate limit:", e);
    }
  };

  const generateWallpaper = async () => {
    if (!prompt.trim() || generating) return;

    setGenerating(true);
    setGeneratedWallpaper(null);

    try {
      const response = await axios.post(`${API}/wallpapers/generate`, { prompt });
      if (response.data.success) {
        setGeneratedWallpaper(response.data.wallpaper);
        setPrompt("");
        await checkRateLimit();
        await fetchTopWallpapers();
      } else {
        alert(response.data.error || "Failed to generate wallpaper");
      }
    } catch (e) {
      if (e.response?.status === 429) {
        alert("Daily generation limit reached (5). Try again tomorrow!");
      } else {
        alert("Error generating wallpaper. Please try again.");
      }
      console.error("Error generating wallpaper:", e);
    } finally {
      setGenerating(false);
    }
  };

  const likeWallpaper = async (wallpaperId) => {
    try {
      const response = await axios.post(`${API}/wallpapers/${wallpaperId}/like`);
      if (response.data.success) {
        await fetchTopWallpapers();
      }
    } catch (e) {
      console.error("Error liking wallpaper:", e);
    }
  };

  const downloadWallpaper = (imageData, prompt) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = `wallpaper-${prompt.substring(0, 30).replace(/[^a-z0-9]/gi, '-')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  useEffect(() => {
    fetchTopWallpapers();
    checkRateLimit();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 text-white">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Sparkles className="w-12 h-12 text-yellow-400" />
            <h1 className="text-6xl font-bold bg-gradient-to-r from-yellow-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
              AI Wallpaper Studio
            </h1>
          </div>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Create stunning, unique wallpapers with the power of AI. Just describe your vision and watch it come to life.
          </p>
        </div>

        {/* How It Works */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 text-center hover:bg-white/20 transition-all">
            <div className="bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              <Wand2 className="w-8 h-8" />
            </div>
            <h3 className="text-2xl font-bold mb-3">1. Describe Your Vision</h3>
            <p className="text-gray-300">Enter a prompt describing the wallpaper you want to create</p>
          </div>

          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 text-center hover:bg-white/20 transition-all">
            <div className="bg-gradient-to-br from-purple-500 to-pink-500 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              <Sparkles className="w-8 h-8" />
            </div>
            <h3 className="text-2xl font-bold mb-3">2. AI Creates Magic</h3>
            <p className="text-gray-300">Our AI generates a unique, high-quality wallpaper just for you</p>
          </div>

          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 text-center hover:bg-white/20 transition-all">
            <div className="bg-gradient-to-br from-pink-500 to-red-500 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              <Download className="w-8 h-8" />
            </div>
            <h3 className="text-2xl font-bold mb-3">3. Download & Enjoy</h3>
            <p className="text-gray-300">Download your wallpaper and set it on any device</p>
          </div>
        </div>

        {/* Generate Section */}
        <div className="max-w-3xl mx-auto mb-16">
          <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl">
            <h2 className="text-3xl font-bold mb-6 text-center">Create Your Wallpaper</h2>
            <div className="flex flex-col gap-4">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe your dream wallpaper... (e.g., 'A serene sunset over mountains with vibrant colors')"
                className="w-full p-4 rounded-xl bg-white/20 border-2 border-white/30 text-white placeholder-gray-400 focus:outline-none focus:border-purple-400 transition-all resize-none h-32"
                disabled={generating}
              />
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-300">
                  {rateLimit.remaining} of {rateLimit.limit} generations remaining today
                </span>
                <button
                  onClick={generateWallpaper}
                  disabled={generating || !prompt.trim() || rateLimit.remaining === 0}
                  className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 px-8 py-4 rounded-xl font-bold text-lg flex items-center gap-2 transition-all transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed"
                >
                  {generating ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-6 h-6" />
                      Generate Wallpaper
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Generated Wallpaper */}
            {generatedWallpaper && (
              <div className="mt-8 animate-fadeIn">
                <h3 className="text-2xl font-bold mb-4 text-center">Your Creation</h3>
                <div className="relative group">
                  <img
                    src={generatedWallpaper.image_data}
                    alt={generatedWallpaper.prompt}
                    className="w-full rounded-xl shadow-2xl"
                  />
                  <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-all rounded-xl flex items-center justify-center gap-4">
                    <button
                      onClick={() => likeWallpaper(generatedWallpaper.id)}
                      className="bg-red-500 hover:bg-red-600 p-4 rounded-full transition-all transform hover:scale-110"
                    >
                      <Heart className="w-6 h-6" />
                    </button>
                    <button
                      onClick={() => downloadWallpaper(generatedWallpaper.image_data, generatedWallpaper.prompt)}
                      className="bg-green-500 hover:bg-green-600 p-4 rounded-full transition-all transform hover:scale-110"
                    >
                      <Download className="w-6 h-6" />
                    </button>
                  </div>
                </div>
                <p className="text-center text-gray-300 mt-4">{generatedWallpaper.prompt}</p>
              </div>
            )}
          </div>
        </div>

        {/* Top 5 Gallery */}
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4 flex items-center justify-center gap-3">
              <ImageIcon className="w-10 h-10 text-pink-400" />
              Top 5 Most Loved Wallpapers
            </h2>
            <p className="text-gray-300">Discover the community's favorite creations</p>
          </div>

          {topWallpapers.length === 0 ? (
            <div className="text-center bg-white/10 backdrop-blur-lg rounded-2xl p-12">
              <p className="text-xl text-gray-400">No wallpapers yet. Be the first to create one!</p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {topWallpapers.map((wallpaper, index) => (
                <div key={wallpaper.id} className="bg-white/10 backdrop-blur-lg rounded-2xl overflow-hidden hover:scale-105 transition-all shadow-xl">
                  <div className="relative">
                    {index === 0 && (
                      <div className="absolute top-4 right-4 bg-yellow-500 text-black px-3 py-1 rounded-full font-bold flex items-center gap-1 z-10">
                        <Sparkles className="w-4 h-4" /> #1
                      </div>
                    )}
                    <img
                      src={wallpaper.image_data}
                      alt={wallpaper.prompt}
                      className="w-full h-64 object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 hover:opacity-100 transition-all flex items-end justify-center pb-4 gap-3">
                      <button
                        onClick={() => likeWallpaper(wallpaper.id)}
                        className="bg-red-500 hover:bg-red-600 p-3 rounded-full transition-all transform hover:scale-110"
                      >
                        <Heart className="w-5 h-5" />
                      </button>
                      <button
                        onClick={() => downloadWallpaper(wallpaper.image_data, wallpaper.prompt)}
                        className="bg-green-500 hover:bg-green-600 p-3 rounded-full transition-all transform hover:scale-110"
                      >
                        <Download className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                  <div className="p-4">
                    <p className="text-sm text-gray-300 line-clamp-2 mb-2">{wallpaper.prompt}</p>
                    <div className="flex items-center gap-2 text-red-400">
                      <Heart className="w-4 h-4 fill-current" />
                      <span className="font-bold">{wallpaper.likes} likes</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-black/30 py-8 mt-16">
        <div className="container mx-auto px-4 text-center text-gray-400">
          <p>© 2025 AI Wallpaper Studio. Powered by AI creativity.</p>
        </div>
      </footer>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />}>
            <Route index element={<Home />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
