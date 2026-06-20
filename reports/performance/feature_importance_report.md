# Feature Importance & Anomaly Explainability Report

This report provides an in-depth analysis of feature significance and model explainability for the **Local Outlier Factor (LOF)** model—the best-performing architecture in the ShieldNet AI network anomaly detection system. 

We explain feature contributions using two complementary methodologies:
1. **Permutation Feature Importance**: A model-agnostic technique that measures the degradation of the model's F1-score when individual feature columns are randomly shuffled.
2. **SHAP (SHapley Additive exPlanations)**: A game-theoretic approach that explains individual predictions by calculating the additive contribution of each network flow feature to the anomaly score.

---

## 1. Top 15 Network Features by Permutation Importance

The table below details the top 15 features that drive the LOF model's anomaly predictions, along with the average decrease in the model's F1-score when the feature's values are randomized:

| Rank | Feature Name | Description | Importance (F1-Score Drop) | Operational Security Context |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `service` | Application-layer service (e.g., HTTP, DNS, SMTP, FTP) | **12.37%** | Anomalies often run on unusual ports or mask traffic under unrecognized service signatures. |
| **2** | `sttl` | Source-to-destination Time-to-Live (TTL) | **10.67%** | Attack packets (e.g., spoofed IPs, custom injection tools) often have abnormal hop limits compared to standard OS defaults. |
| **3** | `proto` | Transaction protocol (e.g., TCP, UDP, ICMP, routing protocols) | **9.39%** | Scanning and flood attacks utilize raw sockets or uncommon protocols to exploit lower-level OS layers. |
| **4** | `ct_state_ttl` | Connections with the same state and TTL | **8.49%** | Indicates high-density state/TTL clustering, characteristic of automated scripting or scanning tools. |
| **5** | `state` | Current state of the network flow (e.g., CON, INT, FIN, REQ) | **5.79%** | Exploits and denial-of-service (DoS) attempts disrupt connection states, leaving flows in half-open states (e.g., INT). |
| **6** | `dwin` | Destination TCP window size advertisement | **5.70%** | TCP window size changes reflect rate limiting, buffering, or session hijacking indicators. |
| **7** | `swin` | Source TCP window size advertisement | **5.17%** | Abnormal window sizes are characteristic of custom TCP packets crafted by exploit toolkits. |
| **8** | `ackdat` | TCP setup time: SYN-ACK to ACK (handshake completion) | **4.73%** | Exploit traffic or slow networks display handshake lags, signaling DoS or server stress. |
| **9** | `dttl` | Destination-to-source Time-to-Live (TTL) | **4.73%** | Path asymmetry and packet anomalies in reply traffic. |
| **10**| `synack` | TCP setup time: SYN to SYN-ACK | **4.71%** | Measures target server responsiveness; high latency indicates target overload during floods. |
| **11**| `tcprtt` | Total TCP connection setup time (RTT) | **4.49%** | Total round-trip setup time; essential for spotting connection-level anomalies. |
| **12**| `dmean` | Mean flow packet size sent by destination | **4.08%** | Identifies anomalous server responses (e.g., data exfiltration or massive shellcode delivery). |
| **13**| `smean` | Mean flow packet size sent by source | **2.84%** | Identifies large outbound payloads, indicative of buffer overflow attempts or command-and-control injections. |
| **14**| `stcpb` | Source TCP base sequence number | **2.57%** | Initial sequence numbers; deviations can reveal TCP session prediction attacks. |
| **15**| `dtcpb` | Destination TCP base sequence number | **2.36%** | Core sequence tracking to identify session-level anomalies. |

---

## 2. SHAP Explainability & Feature Attribution Analysis

SHAP values offer local, instance-level explanations of how specific feature values increase or decrease the anomaly score. Based on the SHAP Summary Plot (`reports/performance/shap_summary_plot.png`), we observe the following key dynamics:

1. **High Impact of Nominal Features (`service`, `proto`)**:
   * Nominal features encoded via the robust preprocessor show strong local attributions. When `service` or `proto` takes on values that are rare or unseen in training (mapped to `'unknown'`), they produce large positive SHAP values, driving the LOF local reachability score up, which flags the flow as anomalous.
2. **TTL Deviations (`sttl`, `dttl`)**:
   * Time-to-Live metrics show sharp clusters in SHAP values. Standard OS network stacks send packets with specific default TTLs (e.g., 64 for Linux, 128 for Windows, 255 for network hardware). Custom attack tools (e.g., Scapy, Nmap) often generate packets with arbitrary TTLs. The model assigns high positive SHAP values to TTLs that fall outside the typical standard clusters.
3. **TCP Handshake Latency (`tcprtt`, `synack`, `ackdat`)**:
   * For normal connections, the round-trip handshake time is low and clustered. In DoS (Denial of Service) attacks or heavy scanning sequences, the handshake setup times diverge. High values of `tcprtt` and `synack` show high positive SHAP values, proving that handshake degradation is a primary trigger for raising severity levels (Low to Critical).
4. **Packet Size Asymmetry (`smean`, `dmean`)**:
   * Large source packet sizes (`smean`) combined with small destination packet sizes (`dmean`) are characteristic of buffer overflow exploits or command injections. SHAP attributes high anomaly scores to flows with skewed packet size distributions.

---

## 3. Operational Deployment Recommendations

* **Feature Selection Optimization**: The top 10 features capture over **70%** of the model's predictive power. In ultra-high-speed network backbones, we can disable tracking for the bottom 30 features to reduce preprocessing overhead, enabling high-throughput wire-speed anomaly detection.
* **Alert Enrichment**: When ShieldNet AI logs an alert to `alerts.csv`, the system should append the top 3 features with the highest SHAP values as "Threat Context Factors". This will tell security analysts *why* the traffic was flagged (e.g., "Flagged due to anomalous TTL combination and TCP handshake delay"), reducing triage time.
