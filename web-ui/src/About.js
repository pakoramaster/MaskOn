import "./App.css";
import { useNavigate } from "react-router-dom";
import { useEffect } from "react";
import Nav from "./Nav";
import { FaGithub, FaLinkedin } from "react-icons/fa";
import bodyImage from "./assets/body_image.png";

function About() {
  const navigate = useNavigate();

  // Set background color to match MainPage
  useEffect(() => {
    const originalBg = document.body.style.background;
    document.body.style.backgroundColor = "#18202b";
    return () => {
      document.body.style.background = originalBg;
    };
  }, []);

  return (
    <div className="min-h-screen flex flex-col justify-center items-center bg-[#18202b]">
      <Nav />
      {/* Header Section */}
      <header className="flex flex-col items-center justify-end text-center h-[150px]">
        <h1 className="text-7xl font-extrabold text-white mb-2 font-[Poppins,Inter,Arial,sans-serif]">
          MaskOn
        </h1>
        <h2 className="text-lg font-normal text-indigo-400 tracking-wide font-[Inter,Arial,sans-serif]">
          Instant background removal, in your browser.
        </h2>
      </header>

      {/* Body Section */}
      <main className="flex-1 flex flex-col items-center mt-12 w-full">
        <div className="w-screen flex justify-center bg-gradient-to-r from-[#202a38] to-[#16202d]">
          <div className="w-full max-w-7xl px-10 py-6 rounded-2xl flex flex-col gap-4 md:flex-row items-start md:gap-10">
            <div className="flex-1 flex flex-col items-start">
              <h2 className="text-4xl font-extrabold text-indigo-400 mb-7 tracking-tight">
                Why MaskOn?
              </h2>
              <ul className="text-left text-base text-[#e3eaf3] mb-7 w-full pl-4 list-disc font-medium font-sans">
                <li className="mb-2">
                  Runs entirely in your browser – your photos and videos never leave your
                  device.
                </li>
                <li className="mb-2">
                  No uploads, no waiting, no privacy concerns.
                </li>
                <li className="mb-2">
                  Instant background removal for content creators, educators,
                  and more.
                </li>
              </ul>
              <p className="text-indigo-400 font-semibold mb-8 text-left font-mono">
                Your privacy is our priority. All processing happens locally,
                and nothing is stored on any server.
              </p>
              <button
                onClick={() => navigate("/main")}
                className="mt-2 bg-indigo-500 text-white font-semibold text-base px-6 py-2 rounded-full cursor-pointer transition-colors duration-200 hover:bg-indigo-400 tracking-wide"
                style={{ fontFamily: "Poppins, Inter, Arial, sans-serif" }}
              >
                Try it now
              </button>
            </div>
            <div className="flex flex-col justify-center items-center md:items-end md:w-auto w-full">
              <img
                src={bodyImage}
                alt="greenscreen phone"
                className="w-80 h-80 object-contain"
              />
            </div>
          </div>
        </div>
      </main>

      {/* Footer Section */}
      <footer className="mt-10 mb-2 text-white text-base tracking-tight flex flex-row items-center justify-center gap-1">
        <a
          href="https://github.com/pakoramaster"
          target="_blank"
          rel="noopener noreferrer"
          className="font-semibold hover:underline flex items-center"
          style={{ textDecoration: "none" }}
        >
          <FaGithub size={24} fill="white" />
        </a>
        <span className="mx-2 text-xl">|</span>
        <a
          href="https://www.linkedin.com/in/hamzatariq55/"
          target="_blank"
          rel="noopener noreferrer"
          className="font-semibold hover:underline flex items-center"
          style={{ textDecoration: "none" }}
        >
          <FaLinkedin size={24} fill="white" />
        </a>
      </footer>
    </div>
  );
}

export default About;
