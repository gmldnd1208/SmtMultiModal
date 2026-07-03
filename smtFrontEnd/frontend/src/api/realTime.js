import { BASE_URL } from "../config";

// 상단 카드 통계 데이터 조회
export async function fetchDashboardStats() {
  const res = await fetch(`${BASE_URL}/api/realtime/stats`);
  if (!res.ok) throw new Error("stats fetch 실패");
  return res.json();
}

// 서버 기동 이후 메모리 버퍼에 쌓인 최근 수신 데이터 일괄 조회 (최대 100건)
export async function fetchRecentItems() {
  const res = await fetch(`${BASE_URL}/api/realtime/recent`);
  if (!res.ok) throw new Error("recent fetch 실패");
  return res.json(); // { items: [...] }
}

// obb_results DB에서 NOR/DEF 모두 포함한 최근 제품 목록 조회
export async function fetchProducts(limit = 100) {
  const res = await fetch(`${BASE_URL}/api/realtime/products?limit=${limit}`);
  if (!res.ok) throw new Error("products fetch 실패");
  return res.json(); // { items: [...] }
}

// PR/SD별 센서 임계값(mean) 조회
export async function fetchThresholds() {
  const res = await fetch(`${BASE_URL}/api/realtime/thresholds`);
  if (!res.ok) throw new Error("thresholds fetch 실패");
  return res.json(); // { pr: { sensor_name: { mean: ... } }, sd: { ... } }
}

// file_name 기준으로 DB inference_results에서 불량 유형 조회
export async function fetchInferenceResult(fileName) {
  const res = await fetch(`${BASE_URL}/api/realtime/inference/${encodeURIComponent(fileName)}`);
  if (!res.ok) throw new Error("inference fetch 실패");
  return res.json(); // { defect_types: [...], accuracy: ... }
}

// file_name 기준으로 sensor_data 컬렉션에서 sensor_importance 조회
export async function fetchSensorImportance(fileName) {
  const res = await fetch(`${BASE_URL}/api/realtime/sensor/${encodeURIComponent(fileName)}`);
  if (!res.ok) throw new Error("sensor fetch 실패");
  return res.json(); // { sensor_importance: { temperature: 0.29, noise: 0.27, ... } }
}

// Kafka 메시지를 SSE로 실시간 수신하는 구독 함수
// onMessage(item) 콜백으로 새 데이터가 들어올 때마다 호출됨
// 반환값인 EventSource를 저장했다가 .close() 로 구독 해제 가능
export function subscribeRealtimeStream(onMessage) {
  const es = new EventSource(`${BASE_URL}/api/realtime/stream`);
  es.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch {
      // JSON 파싱 실패 시 무시
    }
  };
  return es;
}
