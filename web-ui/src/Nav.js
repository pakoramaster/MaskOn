import { FaGithub } from "react-icons/fa";

export default function Nav() {
  return (
    <nav className="w-full flex items-center justify-between px-4 py-3 bg-[#202a38] shadow">
        <a href="/" className="text-white font-extrabold text-3xl tracking-tight font-[Poppins,Inter,Arial,sans-serif]">MaskOn</a>
        <div className="flex gap-6">
            <a href="https://github.com/pakoramaster/MaskOn" target="_blank" rel="noopener noreferrer" className="text-[#61dafb] font-semibold hover:underline transition-colors">
                <FaGithub size={24} fill="white"/>
            </a>
        </div>
    </nav>
  )
}
