"""
K-Means Module (FIXED & OPTIMIZED)
Perbaikan: duplikasi fungsi, efficiency, code reuse
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Import utility functions dari stdbscan_fixed
# Dalam praktik, gunakan: from stdbscan_fixed import haversine_distance, convert_times_to_days, clear_progress_widgets
# Untuk standalone, define ulang di sini

def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine distance (vectorized)"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def convert_times_to_days(times):
    """Konversi waktu ke hari"""
    if isinstance(times[0], (datetime, pd.Timestamp, np.datetime64)):
        times_pd = pd.to_datetime(times)
        base_time = times_pd.min()
        return np.array([(t - base_time).total_seconds() / 86400 for t in times_pd])
    return np.array(times, dtype=float)


def clear_progress_widgets(progress_bar, status_text):
    """Bersihkan progress widgets"""
    progress_bar.empty()
    status_text.empty()


def calculate_max_spatial_spread_efficient(lats, lons):
    """Hitung max spatial spread dengan efisien"""
    if len(lats) < 2:
        return 0
    
    if len(lats) < 100:
        # Full pairwise untuk dataset kecil
        max_dist = 0
        for i in range(len(lats)):
            for j in range(i+1, len(lats)):
                dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                max_dist = max(max_dist, dist)
        return max_dist
    
    # Sampling untuk dataset besar
    centroid_lat = np.mean(lats)
    centroid_lon = np.mean(lons)
    
    distances_to_centroid = haversine_distance(lats, lons, centroid_lat, centroid_lon)
    idx_sorted = np.argsort(distances_to_centroid)
    idx1, idx2 = idx_sorted[-1], idx_sorted[-2]
    
    return haversine_distance(lats[idx1], lons[idx1], lats[idx2], lons[idx2])


# ============= K-MEANS CLASS =============

class SpatioTemporalKMeans:
    """K-Means Clustering (OPTIMIZED)"""
    
    def __init__(self, n_clusters=3, use_depth=False, spatial_weight=1.0, 
                 temporal_weight=1.0, depth_weight=1.0, random_state=42):
        self.n_clusters = n_clusters
        self.use_depth = use_depth
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.depth_weight = depth_weight
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.scaler = StandardScaler()
        self.inertia_ = None
        
    def _prepare_features(self, latitudes, longitudes, times, depths=None):
        """Persiapan fitur dengan normalisasi"""
        times = convert_times_to_days(times)
        
        # Konversi ke km
        mean_lat = np.mean(latitudes)
        lat_km = np.array(latitudes) * 111.0
        lon_km = np.array(longitudes) * 111.0 * np.cos(np.radians(mean_lat))
        
        # Stack features dengan weights
        if self.use_depth and depths is not None:
            depths = np.array(depths, dtype=float)
            features = np.column_stack([
                lat_km * self.spatial_weight,
                lon_km * self.spatial_weight,
                times * self.temporal_weight,
                depths * self.depth_weight
            ])
        else:
            features = np.column_stack([
                lat_km * self.spatial_weight,
                lon_km * self.spatial_weight,
                times * self.temporal_weight
            ])
        
        return features
    
    def fit(self, latitudes, longitudes, times, depths=None, debug=False):
        """Melakukan klasterisasi K-Means"""
        if self.use_depth and depths is None:
            raise ValueError("Parameter 'depths' harus diisi jika use_depth=True")
        
        features = self._prepare_features(latitudes, longitudes, times, depths)
        
        if debug:
            st.write(f"\n**Debug - Informasi Fitur:**")
            st.write(f"- Mode: {'4D (Spasial-Temporal-Depth)' if self.use_depth else '3D (Spasial-Temporal)'}")
            st.write(f"- Jumlah samples: {len(features)}")
            st.write(f"- Jumlah fitur: {features.shape[1]}")
            st.write(f"- Jumlah cluster: {self.n_clusters}")
        
        # Normalisasi
        features_normalized = self.scaler.fit_transform(features)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing K-Means clustering...")
            progress_bar.progress(0.3)
            
            status_text.text("Running K-Means algorithm...")
            
            kmeans = SKLearnKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            
            self.labels_ = kmeans.fit_predict(features_normalized)
            self.cluster_centers_ = kmeans.cluster_centers_
            self.inertia_ = kmeans.inertia_
            
            progress_bar.progress(1.0)
            mode_text = "4D (Spasial-Temporal-Depth)" if self.use_depth else "3D (Spasial-Temporal)"
            status_text.text(f"✅ K-Means clustering completed ({mode_text})! Found {self.n_clusters} clusters")
            
            if debug:
                unique, counts = np.unique(self.labels_, return_counts=True)
                st.write(f"\n**Distribusi Cluster:**")
                for cluster_id, count in zip(unique, counts):
                    st.write(f"- Cluster {cluster_id}: {count} events ({count/len(self.labels_)*100:.1f}%)")
                st.write(f"\n**Inertia (WCSS):** {self.inertia_:.2f}")
        
        finally:
            import time
            time.sleep(2)
            clear_progress_widgets(progress_bar, status_text)
        
        return self.labels_
    
    def predict(self, latitudes, longitudes, times, depths=None):
        """Prediksi cluster untuk data baru"""
        if self.cluster_centers_ is None:
            raise ValueError("Model belum di-fit. Jalankan fit() terlebih dahulu.")
        
        features = self._prepare_features(latitudes, longitudes, times, depths)
        features_normalized = self.scaler.transform(features)
        
        # Hitung jarak ke semua centroid
        distances = np.linalg.norm(
            features_normalized[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
            axis=2
        )
        
        return np.argmin(distances, axis=1)


# ============= OPTIMAL K FINDER =============

def find_optimal_k(df, k_range=range(2, 11), use_depth=False, 
                   spatial_weight=1.0, temporal_weight=1.0, depth_weight=1.0):
    """Mencari k optimal dengan Elbow Method"""
    inertias = []
    silhouettes = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, k in enumerate(k_range):
            status_text.text(f"Testing k={k}...")
            progress_bar.progress((i + 1) / len(k_range))
            
            model = SpatioTemporalKMeans(
                n_clusters=k,
                use_depth=use_depth,
                spatial_weight=spatial_weight,
                temporal_weight=temporal_weight,
                depth_weight=depth_weight
            )
            
            if use_depth:
                labels = model.fit(
                    df['latitude'].values,
                    df['longitude'].values,
                    df['datetime'].values,
                    depths=df['depth'].values
                )
            else:
                labels = model.fit(
                    df['latitude'].values,
                    df['longitude'].values,
                    df['datetime'].values
                )
            
            inertias.append(model.inertia_)
            
            # Silhouette score
            features = model._prepare_features(
                df['latitude'].values,
                df['longitude'].values,
                df['datetime'].values,
                df['depth'].values if use_depth else None
            )
            features_normalized = model.scaler.transform(features)
            
            try:
                silhouette = silhouette_score(features_normalized, labels)
                silhouettes.append(silhouette)
            except:
                silhouettes.append(0)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Optimal k analysis completed!")
        
    finally:
        import time
        time.sleep(2)
        clear_progress_widgets(progress_bar, status_text)
    
    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes
    }


# ============= METRICS =============

def calculate_kmeans_metrics(df, labels, model):
    """Menghitung metrik evaluasi K-Means"""
    n_clusters = len(np.unique(labels))
    
    if n_clusters < 2:
        return None
    
    features = model._prepare_features(
        df['latitude'].values,
        df['longitude'].values,
        df['datetime'].values,
        df['depth'].values if model.use_depth else None
    )
    features_normalized = model.scaler.transform(features)
    
    try:
        silhouette = silhouette_score(features_normalized, labels, metric='euclidean')
    except:
        silhouette = None
    
    try:
        davies_bouldin = davies_bouldin_score(features_normalized, labels)
    except:
        davies_bouldin = None
    
    unique, counts = np.unique(labels, return_counts=True)
    avg_cluster_size = np.mean(counts)
    cluster_balance = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'inertia': model.inertia_,
        'n_clusters': n_clusters,
        'avg_cluster_size': avg_cluster_size,
        'cluster_balance': cluster_balance,
        'cluster_sizes': dict(zip(unique, counts))
    }


# ============= PATTERN CLASSIFICATION =============

def classify_kmeans_cluster_pattern(cluster_data):
    """Klasifikasi pola cluster K-Means (OPTIMIZED)"""
    durasi = (cluster_data['datetime'].max() - cluster_data['datetime'].min()).days
    
    lats = cluster_data['latitude'].values
    lons = cluster_data['longitude'].values
    
    # Efficient spatial spread
    max_spatial_spread = calculate_max_spatial_spread_efficient(lats, lons)
    
    # Mean spatial spread (dengan sampling untuk dataset besar)
    if len(lats) < 50:
        spatial_distances = []
        for i in range(len(lats)):
            for j in range(i+1, len(lats)):
                dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                spatial_distances.append(dist)
        mean_spatial_spread = np.mean(spatial_distances) if spatial_distances else 0
    else:
        # Sampling
        sample_size = min(50, len(lats))
        sample_idx = np.random.choice(len(lats), sample_size, replace=False)
        sample_lats = lats[sample_idx]
        sample_lons = lons[sample_idx]
        
        spatial_distances = []
        for i in range(len(sample_lats)):
            for j in range(i+1, len(sample_lats)):
                dist = haversine_distance(sample_lats[i], sample_lons[i], 
                                         sample_lats[j], sample_lons[j])
                spatial_distances.append(dist)
        mean_spatial_spread = np.mean(spatial_distances) if spatial_distances else 0
    
    # Time gaps
    times_sorted = cluster_data['datetime'].sort_values()
    time_gaps = [(times_sorted.iloc[i+1] - times_sorted.iloc[i]).days 
                 for i in range(len(times_sorted)-1)]
    
    max_time_gap = max(time_gaps) if time_gaps else 0
    mean_time_gap = np.mean(time_gaps) if time_gaps else 0
    
    # Movement
    cluster_sorted = cluster_data.sort_values('datetime')
    n = len(cluster_sorted)
    
    total_movement = 0
    if n >= 3:
        seg_size = n // 3
        segs = [
            cluster_sorted.iloc[:seg_size],
            cluster_sorted.iloc[seg_size:2*seg_size],
            cluster_sorted.iloc[2*seg_size:]
        ]
        
        centroids = [(s['latitude'].mean(), s['longitude'].mean()) for s in segs]
        
        total_movement = sum(
            haversine_distance(centroids[i][0], centroids[i][1], 
                             centroids[i+1][0], centroids[i+1][1])
            for i in range(len(centroids)-1)
        )
    
    temporal_intensity = len(cluster_data) / max(durasi, 1)
    
    # Classification
    pattern = "UNDEFINED"
    confidence = 0
    description = ""
    
    if durasi < 15:
        pattern = "OCCASIONAL"
        confidence = 90
        description = f"Burst gempa dalam waktu singkat ({durasi} hari, intensitas: {temporal_intensity:.2f} events/hari)"
    elif durasi > 30 and max_spatial_spread < 50 and max_time_gap < 30:
        pattern = "STATIONARY"
        confidence = 85
        description = f"Aktivitas kontinu di lokasi tetap (durasi={durasi} hari, spread={max_spatial_spread:.1f} km)"
    elif max_time_gap > 30 and mean_spatial_spread < 40:
        pattern = "REAPPEARING"
        confidence = 80
        description = f"Aktivitas berulang dengan gap maksimum {max_time_gap} hari"
    elif total_movement > 30 and durasi > 15:
        pattern = "TRACK"
        confidence = 75
        description = f"Pergerakan spatial {total_movement:.1f} km selama {durasi} hari"
    elif max_spatial_spread > 100 and durasi > 30 and temporal_intensity < 0.5:
        pattern = "DISPERSED"
        confidence = 70
        description = f"Aktivitas tersebar luas (spread={max_spatial_spread:.1f} km, low intensity)"
    elif max_spatial_spread < 30 and temporal_intensity > 1.0:
        pattern = "COMPACT"
        confidence = 75
        description = f"Cluster kompak dan aktif (spread={max_spatial_spread:.1f} km, {temporal_intensity:.2f} events/hari)"
    else:
        pattern = "STATIONARY" if durasi >= 30 else "OCCASIONAL"
        confidence = 60
        description = f"Aktivitas {'berkepanjangan' if durasi >= 30 else 'singkat'} (durasi={durasi} hari)"
    
    return {
        'pattern': pattern,
        'confidence': confidence,
        'description': description,
        'durasi': durasi,
        'max_spatial_spread': max_spatial_spread,
        'mean_spatial_spread': mean_spatial_spread,
        'max_time_gap': max_time_gap,
        'mean_time_gap': mean_time_gap,
        'total_movement': total_movement,
        'temporal_intensity': temporal_intensity,
        'n_events': len(cluster_data)
    }


# ============= CLUSTER ANALYSIS =============

def analyze_kmeans_clusters_with_patterns(df, labels):
    """
    Analisis lengkap per cluster K-Means dengan pola
    INI ADALAH FUNGSI UTAMA - yang lain dihapus untuk menghindari duplikasi
    """
    df_analysis = df.copy()
    df_analysis['cluster'] = labels
    
    n_clusters = len(np.unique(labels))
    
    stats = []
    patterns = []
    
    for cluster_id in sorted(np.unique(labels)):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        
        # Pattern classification
        pattern_info = classify_kmeans_cluster_pattern(cluster_data)
        
        # Spatial spread (efficient calculation)
        lats = cluster_data['latitude'].values
        lons = cluster_data['longitude'].values
        max_dist = calculate_max_spatial_spread_efficient(lats, lons)
        
        # Centroid
        centroid_lat = cluster_data['latitude'].mean()
        centroid_lon = cluster_data['longitude'].mean()
        
        stats.append({
            'Cluster': cluster_id,
            'Jumlah Events': len(cluster_data),
            'Centroid Lat': centroid_lat,
            'Centroid Lon': centroid_lon,
            'Mag Min': cluster_data['magnitude'].min(),
            'Mag Max': cluster_data['magnitude'].max(),
            'Mag Mean': cluster_data['magnitude'].mean(),
            'Depth Min (km)': cluster_data['depth'].min(),
            'Depth Max (km)': cluster_data['depth'].max(),
            'Depth Mean (km)': cluster_data['depth'].mean(),
            'Durasi (hari)': pattern_info['durasi'],
            'Spatial Spread (km)': max_dist,
            'Pola': pattern_info['pattern'],
            'Confidence (%)': pattern_info['confidence']
        })
        
        patterns.append({
            'cluster_id': cluster_id,
            'pattern_info': pattern_info
        })
    
    return pd.DataFrame(stats), n_clusters, patterns


# Backward compatibility - redirect ke fungsi utama
def analyze_kmeans_clusters(df, labels):
    """
    Wrapper untuk backward compatibility
    Mengembalikan stats tanpa patterns
    """
    stats_df, n_clusters, _ = analyze_kmeans_clusters_with_patterns(df, labels)
    return stats_df, n_clusters
