import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.aistudio.app',
  appName: 'AI Studio',
  webDir: 'www',
  server: {
    androidScheme: 'https',
    // Allow connections to local LLM server and RunPod
    cleartext: true,
  },
  android: {
    allowMixedContent: true,
  },
};

export default config;
