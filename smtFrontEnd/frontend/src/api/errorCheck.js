import { BASE_URL } from "../config";

export async function fetchErrorItems(status = null, limit = 50) {
  const params = new URLSearchParams({ limit });
  if (status) params.append("status", status);
  const res = await fetch(`${BASE_URL}/api/errorcheck/items?${params}`);
  if (!res.ok) throw new Error("items fetch 실패");
  return res.json();
}

export async function fetchErrorStats() {
  const res = await fetch(`${BASE_URL}/api/errorcheck/stats`);
  if (!res.ok) throw new Error("stats fetch 실패");
  return res.json();
}

export async function approveItem(id) {
  const res = await fetch(`${BASE_URL}/api/errorcheck/items/${id}/approve`, { method: "PATCH" });
  if (!res.ok) throw new Error("approve 실패");
  return res.json();
}

export async function rejectItem(id) {
  const res = await fetch(`${BASE_URL}/api/errorcheck/items/${id}/reject`, { method: "PATCH" });
  if (!res.ok) throw new Error("reject 실패");
  return res.json();
}
